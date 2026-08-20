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
        description="Obtaining video IDs and metadata from search words",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "wordlist",
        type=str,
        help="filename of word list"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="videoid",
        help="dirname to save video IDs"
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=cpu_count(),
        help="Number of parallel processes to use"
    )
    return parser.parse_args(sys.argv[1:])


def process_word(word):
    try:
        # Download search results
        url = make_query_url(word)
        html = requests.get(url).content
        text = str(html)

        results = []
        seen = set()

        # Find every videoId
        video_matches = re.finditer(
            r'"videoId":"([\w_-]+)"',
            text
        )

        for match in video_matches:
            video_id = match.group(1)

            # Avoid duplicates
            if video_id in seen:
                continue

            seen.add(video_id)

            # Only inspect a limited section after the videoId.
            # This is much faster than running .*? across the
            # entire HTML document.
            chunk = text[match.start():match.start() + 5000]

            # Channel name + channel ID
            channel_match = re.search(
                r'"longBylineText":\{"runs":\[\{"text":"(.*?)".*?'
                r'"browseEndpoint":\{"browseId":"(UC[\w_-]+)"',
                chunk
            )

            # Duration
            duration_match = re.search(
                r'"lengthText":\{.*?"simpleText":"([^"]+)"',
                chunk
            )

            # If the expected metadata wasn't found, skip this video
            if not channel_match or not duration_match:
                continue

            channel_name, channel_id = channel_match.groups()
            duration = duration_match.group(1)

            results.append({
                "video_id": video_id,
                "channel_id": channel_id,
                "channel_name": channel_name,
                "duration": duration,
            })

        return word, results

    except Exception as e:
        print(f"No video found for {word}: {e}")
        return word, []


def obtain_video_id(fn_word, outdir, processes):
    fn_videoid = Path(outdir) / f"{Path(fn_word).stem}.csv"
    fn_videoid.parent.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # Find words that have already been processed
    # ---------------------------------------------------------
    processed_words = set()

    if fn_videoid.exists():
        with open(
            fn_videoid,
            "r",
            newline="",
            encoding="utf-8"
        ) as f:
            reader = csv.reader(f)

            # Skip header
            next(reader, None)

            for row in reader:
                if row:
                    processed_words.add(row[0])

    # ---------------------------------------------------------
    # Read word list
    # ---------------------------------------------------------
    with open(fn_word, encoding="utf-8") as f:
        words = [
            w.strip()
            for w in f.readlines()
            if w.strip()
        ]

    words_to_process = [
        w for w in words
        if w not in processed_words
    ]

    if not words_to_process:
        print("All words already processed!")
        return fn_videoid

    # ---------------------------------------------------------
    # Process words in parallel
    # ---------------------------------------------------------
    with Pool(processes) as pool:

        for word, results in tqdm(
            pool.imap_unordered(
                process_word,
                words_to_process
            ),
            total=len(words_to_process)
        ):

            with open(
                fn_videoid,
                "a",
                newline="",
                encoding="utf-8"
            ) as f:

                writer = csv.writer(f)

                # Write header if file is empty
                if f.tell() == 0:
                    writer.writerow([
                        "word",
                        "video_id",
                        "channel_id",
                        "channel_name",
                        "duration",
                        "video_link"
                    ])

                # Write results
                for result in results:

                    video_id = result["video_id"]

                    video_link = (
                        f"https://www.youtube.com/watch?v={video_id}"
                    )

                    writer.writerow([
                        word,
                        video_id,
                        result["channel_id"],
                        result["channel_name"],
                        result["duration"],
                        video_link
                    ])

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