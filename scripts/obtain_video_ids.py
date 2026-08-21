import argparse
import csv
import re
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path

import requests
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm

from scripts.utils import make_query_url


def parse_args():
    parser = argparse.ArgumentParser(
        description="Obtaining video IDs from search words",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "wordlist",
        type=str,
        help="filename of word list",
    )

    parser.add_argument(
        "--outdir",
        type=str,
        default="videoid",
        help="dirname to save video IDs",
    )

    parser.add_argument(
        "--processes",
        type=int,
        default=cpu_count(),
        help="Number of parallel processes to use",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=2000,
        help="Number of words processed before saving and uploading a checkpoint",
    )

    parser.add_argument(
        "--hf_dataset_repo",
        type=str,
        default=None,
        help="Hugging Face dataset repository, e.g. username/dataset-name",
    )

    return parser.parse_args(sys.argv[1:])


def process_word(word):
    try:
        # Download search results
        url = make_query_url(word)
        html = requests.get(url, timeout=30).content

        # Find video IDs
        videoids_found = [
            x.split(":")[1].strip('"').strip(" ")
            for x in re.findall(
                r'"videoId":"[\w_\-]+?"',
                str(html),
            )
        ]

        return word, list(set(videoids_found))

    except Exception as e:
        print(f"No video found for {word}: {e}")
        return word, []


def download_hf_csv(repo_id, filename, local_path):
    """
    Download the existing CSV checkpoint from a Hugging Face dataset repo.

    Returns True if a remote CSV was found, otherwise False.
    """
    try:
        downloaded_file = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
        )

        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy the downloaded file to our local output location.
        with open(downloaded_file, "rb") as src, open(local_path, "wb") as dst:
            dst.write(src.read())

        print(
            f"Downloaded existing checkpoint from "
            f"https://huggingface.co/datasets/{repo_id}/blob/main/{filename}"
        )

        return True

    except Exception as e:
        print(f"No existing Hugging Face checkpoint found: {e}")
        return False


def upload_hf_csv(repo_id, local_path):
    """
    Upload the local CSV to the Hugging Face dataset repository.
    """
    api = HfApi()

    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=local_path.name,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Update {local_path.name}",
    )

    print(
        f"Uploaded checkpoint to "
        f"https://huggingface.co/datasets/{repo_id}/blob/main/{local_path.name}"
    )


def load_processed_words(csv_path):
    """
    Read the checkpoint CSV and return the set of words already processed.
    """
    processed_words = set()

    if not csv_path.exists():
        return processed_words

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)

        # Skip header
        next(reader, None)

        for row in reader:
            if row:
                processed_words.add(row[0])

    return processed_words


def append_results(csv_path, results):
    """
    Append a batch of results to the CSV.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = csv_path.exists() and csv_path.stat().st_size > 0

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["word", "video_id", "video_link"])

        for word, videoids in results:
            for videoid in videoids:
                video_link = f"https://www.youtube.com/watch?v={videoid}"
                writer.writerow([word, videoid, video_link])

        f.flush()


def obtain_video_id(
    fn_word,
    outdir,
    processes,
    batch_size,
    hf_dataset_repo=None,
):
    fn_videoid = Path(outdir) / f"{Path(fn_word).stem}.csv"
    fn_videoid.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Fetch the latest checkpoint from Hugging Face at the beginning
    # of every run.
    # ------------------------------------------------------------------
    hf_checkpoint_exists = False

    if hf_dataset_repo:
        hf_checkpoint_exists = download_hf_csv(
            hf_dataset_repo,
            fn_videoid.name,
            fn_videoid,
        )

    # ------------------------------------------------------------------
    # Read already processed words from the checkpoint.
    # ------------------------------------------------------------------
    processed_words = load_processed_words(fn_videoid)

    print(f"Already processed: {len(processed_words)} words")

    # ------------------------------------------------------------------
    # Read word list.
    # ------------------------------------------------------------------
    with open(fn_word, "r", encoding="utf-8") as f:
        words = [w.strip() for w in f if w.strip()]

    words_to_process = [
        word for word in words
        if word not in processed_words
    ]

    print(f"Total words:       {len(words)}")
    print(f"Remaining words:   {len(words_to_process)}")
    print(f"Batch size:        {batch_size}")

    if not words_to_process:
        print("All words already processed!")
        return fn_videoid

    # ------------------------------------------------------------------
    # Process words in parallel.
    #
    # Results are buffered until batch_size words have completed.
    # Then:
    #   1. Save them to CSV
    #   2. Upload CSV to HF
    #
    # This means an interrupted run can lose at most one incomplete
    # batch.
    # ------------------------------------------------------------------
    batch = []
    completed_since_checkpoint = 0

    with Pool(processes) as pool:
        iterator = pool.imap_unordered(
            process_word,
            words_to_process,
        )

        for word, videoids in tqdm(
            iterator,
            total=len(words_to_process),
            desc="Processing",
        ):
            batch.append((word, videoids))
            completed_since_checkpoint += 1

            # ----------------------------------------------------------
            # Checkpoint every batch_size words.
            # ----------------------------------------------------------
            if completed_since_checkpoint >= batch_size:
                append_results(fn_videoid, batch)

                print(
                    f"\nCheckpoint reached: "
                    f"{len(batch)} words"
                )

                if hf_dataset_repo:
                    upload_hf_csv(
                        hf_dataset_repo,
                        fn_videoid,
                    )

                batch = []
                completed_since_checkpoint = 0

    # ------------------------------------------------------------------
    # Handle the final partial batch.
    #
    # On the FIRST run, if there was never a 2000-word checkpoint,
    # don't upload a smaller CSV. This implements:
    #
    #   "If it's the first time wait until the first batch size to upload."
    #
    # However, once an HF checkpoint already exists, upload the final
    # partial batch so that resumed progress isn't lost.
    # ------------------------------------------------------------------
    if batch:
        append_results(fn_videoid, batch)

        if hf_dataset_repo:
            if hf_checkpoint_exists:
                print(
                    f"\nUploading final partial batch "
                    f"({len(batch)} words) to Hugging Face..."
                )

                upload_hf_csv(
                    hf_dataset_repo,
                    fn_videoid,
                )
            else:
                print(
                    f"\nFinal batch contains only {len(batch)} words. "
                    f"Not uploading because this is the first checkpoint."
                )

    return fn_videoid


if __name__ == "__main__":
    args = parse_args()

    filename = obtain_video_id(
        args.wordlist,
        args.outdir,
        args.processes,
        args.batch_size,
        args.hf_dataset_repo,
    )

    print(f"Saved video IDs to {filename}.")