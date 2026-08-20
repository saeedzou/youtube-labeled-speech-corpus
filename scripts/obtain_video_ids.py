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
        description="Obtaining video IDs from YouTube search words",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "wordlist",
        type=str,
        help="Filename of word list"
    )

    parser.add_argument(
        "--outdir",
        type=str,
        default="videoid",
        help="Directory to save video IDs"
    )

    parser.add_argument(
        "--processes",
        type=int,
        default=cpu_count(),
        help="Number of parallel processes to use"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of completed words to batch before writing to CSV"
    )

    return parser.parse_args(sys.argv[1:])


def process_word(word):
    try:
        # Download search results
        url = make_query_url(word)
        html = requests.get(url).content
        text = str(html)

        # Find video ID, channel name, and channel ID
        pattern = (
            r'"videoId":"([\w_-]+)"'
            r'.*?'
            r'"longBylineText":\{"runs":\[\{"text":"(.*?)".*?'
            r'"browseEndpoint":\{"browseId":"(UC[\w_-]+)"'
        )

        matches = re.findall(pattern, text)

        # Remove duplicate video IDs
        seen = set()
        results = []

        for video_id, channel_name, channel_id in matches:
            if video_id in seen:
                continue

            seen.add(video_id)

            results.append({
                "video_id": video_id,
                "channel_id": channel_id,
                "channel_name": channel_name,
            })

        return word, results

    except Exception as e:
        print(f"No video found for {word}: {e}")
        return word, []


def write_batch(fn_videoid, batch):
    """
    Append a batch of processed words to the CSV.
    """
    file_exists = fn_videoid.exists()
    file_empty = not file_exists or fn_videoid.stat().st_size == 0

    with open(
        fn_videoid,
        "a",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        if file_empty:
            writer.writerow([
                "word",
                "video_id",
                "channel_id",
                "channel_name",
                "video_link"
            ])

        for word, results in batch:
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
                    video_link
                ])

        f.flush()


def obtain_video_id(fn_word, outdir, processes, batch_size):
    fn_videoid = Path(outdir) / f"{Path(fn_word).stem}.csv"
    fn_videoid.parent.mkdir(parents=True, exist_ok=True)

    # Determine which words have already been processed
    processed_words = set()

    if fn_videoid.exists():
        with open(
            fn_videoid,
            "r",
            newline="",
            encoding="utf-8"
        ) as f:

            reader = csv.reader(f)
            next(reader, None)

            for row in reader:
                if row:
                    processed_words.add(row[0])

    # Read word list
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

    print(f"Total words: {len(words)}")
    print(f"Already processed: {len(processed_words)}")
    print(f"Remaining: {len(words_to_process)}")
    print(f"Processes: {processes}")
    print(f"Batch size: {batch_size}")

    batch = []

    with Pool(processes) as pool:
        for word, results in tqdm(
            pool.imap_unordered(
                process_word,
                words_to_process
            ),
            total=len(words_to_process)
        ):
            batch.append((word, results))

            # Write once the batch reaches the requested size
            if len(batch) >= batch_size:
                write_batch(fn_videoid, batch)
                batch.clear()

    # Write any remaining results
    if batch:
        write_batch(fn_videoid, batch)
        batch.clear()

    return fn_videoid


if __name__ == "__main__":
    args = parse_args()

    filename = obtain_video_id(
        args.wordlist,
        args.outdir,
        args.processes,
        args.batch_size
    )

    print(f"Saved video IDs to {filename}.")