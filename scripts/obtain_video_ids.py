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


def obtain_video_id(fn_word, outdir, processes):
    fn_videoid = Path(outdir) / f"{Path(fn_word).stem}.csv"
    fn_videoid.parent.mkdir(parents=True, exist_ok=True)

    # Read the word list and slice the batch
    words = list(open(fn_word, "r").readlines())
    words = [word.strip() for word in words]

    # Open the output CSV file
    with open(fn_videoid, "a", newline="") as f:
        writer = csv.writer(f)
        # Write header if the file is empty
        if f.tell() == 0:
            writer.writerow(["word", "video_id", "video_link"])

        # Process words and write results as they become available
        with Pool(processes) as pool:
            for word, videoids in tqdm(pool.imap_unordered(process_word, words), total=len(words)):
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
