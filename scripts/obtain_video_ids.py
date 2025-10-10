import time
import requests
import argparse
import re
import sys
from pathlib import Path
from scripts.utils import make_query_url
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import csv


def parse_args():
    parser = argparse.ArgumentParser(
        description="Obtaining video IDs from search words",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("wordlist", type=str, help="filename of word list")
    parser.add_argument("--outdir", type=str, default="videoid", help="dirname to save video IDs")
    parser.add_argument("--processes", type=int, default=cpu_count(), help="Number of parallel processes to use")
    return parser.parse_args(sys.argv[1:])


def process_word(word):
    try:
        # Download search results
        url = make_query_url(word)
        html = requests.get(url).content

        # Find video IDs
        videoids_found = [x.split(":")[1].strip("\"").strip(" ") for x in re.findall(r"\"videoId\":\"[\w\_\-]+?\"", str(html))]
        return word, list(set(videoids_found))
    except Exception:
        print(f"No video found for {word}.")
        return word, []


# Instead of one open() for the whole run, do:
def obtain_video_id(fn_word, outdir, processes):
    fn_videoid = Path(outdir) / f"{Path(fn_word).stem}.csv"
    fn_videoid.parent.mkdir(parents=True, exist_ok=True)

    processed_words = set()
    if fn_videoid.exists():
        with open(fn_videoid, "r", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                processed_words.add(row[0])

    words = [w.strip() for w in open(fn_word).readlines()]
    words_to_process = [w for w in words if w not in processed_words]

    if not words_to_process:
        print("All words already processed!")
        return fn_videoid

    with Pool(processes) as pool:
        for word, videoids in tqdm(pool.imap_unordered(process_word, words_to_process), total=len(words_to_process)):
            with open(fn_videoid, "a", newline="") as f:
                writer = csv.writer(f)
                if f.tell() == 0:
                    writer.writerow(["word", "video_id", "video_link"])
                for videoid in videoids:
                    video_link = f"https://www.youtube.com/watch?v={videoid}"
                    writer.writerow([word, videoid, video_link])
                f.flush()
    
    return fn_videoid



if __name__ == "__main__":
    args = parse_args()

    filename = obtain_video_id(
        args.wordlist,
        args.outdir,
        args.processes
    )
    print(f"Saved video IDs to {filename}.")
