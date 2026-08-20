import argparse
import csv
import os
import re
import shutil
from multiprocessing import Pool, cpu_count
from pathlib import Path

import requests
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from tqdm import tqdm

from scripts.utils import make_query_url


DEFAULT_BATCH_SIZE = 2000


def parse_args():
    parser = argparse.ArgumentParser(
        description="Obtain YouTube video IDs and channel information from search words",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "wordlist",
        type=str,
        help="Filename of word list",
    )

    parser.add_argument(
        "--outdir",
        type=str,
        default="videoid",
        help="Directory to save video IDs",
    )

    parser.add_argument(
        "--processes",
        type=int,
        default=cpu_count(),
        help="Number of parallel processes to use",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of completed words before saving/uploading",
    )

    parser.add_argument(
        "--hf_dataset_repo",
        type=str,
        default=None,
        help="Hugging Face dataset repository, e.g. username/my-dataset",
    )

    return parser.parse_args()


def process_word(word):
    """
    Search YouTube for a word and extract:

        video_id
        channel_id
        channel_name
    """

    try:
        url = make_query_url(word)

        response = requests.get(
            url,
            timeout=30,
        )

        response.raise_for_status()

        text = str(response.content)

        pattern = (
            r'"videoId":"([\w_-]+)"'
            r'.*?'
            r'"longBylineText":\{"runs":\[\{"text":"(.*?)".*?'
            r'"browseEndpoint":\{"browseId":"(UC[\w_-]+)"'
        )

        matches = re.findall(
            pattern,
            text,
        )

        # Deduplicate videos while preserving order.
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
        print(
            f"No video found for {word}: {e}"
        )

        return word, []


def get_hf_token():
    """
    Get the Hugging Face token from the environment.
    """

    token = os.environ.get("HF_TOKEN")

    if not token:
        raise RuntimeError(
            "HF_TOKEN is not set. "
            "Please set your Hugging Face token in the environment."
        )

    return token


def download_from_huggingface(
    repo_id,
    filename,
    local_path,
):
    """
    Try to download an existing CSV from the HF dataset.

    Returns:
        True  -> file was downloaded
        False -> file does not exist yet
    """

    print(
        f"Checking Hugging Face for {filename}..."
    )

    token = get_hf_token()

    try:

        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            token=token,
        )

        local_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.copy2(
            downloaded_path,
            local_path,
        )

        print(
            f"Downloaded existing CSV:"
            f" {repo_id}/{filename}"
        )

        return True

    except EntryNotFoundError:

        print(
            f"{filename} does not exist in "
            f"{repo_id}."
        )

        return False

    except Exception as e:

        # If the repo is completely empty, depending on the
        # HF Hub version, this can also appear as a 404.
        error_text = str(e)

        if "404" in error_text or "Not Found" in error_text:

            print(
                f"Could not find {filename} in "
                f"{repo_id}."
            )

            return False

        raise


def upload_to_huggingface(
    repo_id,
    local_path,
):
    """
    Create the HF dataset repository if necessary
    and upload the current CSV.
    """

    token = get_hf_token()

    api = HfApi(
        token=token,
    )

    # Create the dataset repository if it does not exist.
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        exist_ok=True,
    )

    filename = local_path.name

    print(
        f"Uploading {filename} "
        f"to {repo_id}..."
    )

    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=filename,
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
    )

    print(
        f"Uploaded {filename} "
        f"to {repo_id}"
    )


def get_processed_words(
    csv_path,
):
    """
    Return the set of words already present in the CSV.
    """

    processed_words = set()

    if not csv_path.exists():
        return processed_words

    with open(
        csv_path,
        "r",
        newline="",
        encoding="utf-8",
    ) as f:

        reader = csv.reader(f)

        # Skip header.
        next(reader, None)

        for row in reader:

            if row:
                processed_words.add(
                    row[0]
                )

    return processed_words


def write_batch(
    csv_path,
    batch,
):
    """
    Append a batch of results to the CSV.
    """

    file_empty = (
        not csv_path.exists()
        or csv_path.stat().st_size == 0
    )

    with open(
        csv_path,
        "a",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        if file_empty:

            writer.writerow([
                "word",
                "video_id",
                "channel_id",
                "channel_name",
                "video_link",
            ])

        for word, results in batch:

            for result in results:

                video_id = result[
                    "video_id"
                ]

                video_link = (
                    "https://www.youtube.com/watch?v="
                    + video_id
                )

                writer.writerow([
                    word,
                    video_id,
                    result["channel_id"],
                    result["channel_name"],
                    video_link,
                ])

        f.flush()


def obtain_video_id(
    fn_word,
    outdir,
    processes,
    batch_size,
    hf_dataset_repo=None,
):
    fn_word = Path(fn_word)

    csv_path = (
        Path(outdir)
        / f"{fn_word.stem}.csv"
    )

    csv_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =========================================================
    # Hugging Face initialization
    # =========================================================

    if hf_dataset_repo:

        if csv_path.exists():

            print(
                f"Using existing local CSV:"
                f" {csv_path}"
            )

        else:

            downloaded = (
                download_from_huggingface(
                    hf_dataset_repo,
                    csv_path.name,
                    csv_path,
                )
            )

            if not downloaded:

                print(
                    "No existing CSV found. "
                    "Starting from scratch."
                )

    # =========================================================
    # Load already processed words
    # =========================================================

    processed_words = get_processed_words(
        csv_path
    )

    # =========================================================
    # Load word list
    # =========================================================

    with open(
        fn_word,
        "r",
        encoding="utf-8",
    ) as f:

        words = [
            line.strip()
            for line in f
            if line.strip()
        ]

    # Remove duplicates from input while preserving order.
    words = list(
        dict.fromkeys(words)
    )

    words_to_process = [
        word
        for word in words
        if word not in processed_words
    ]

    # =========================================================
    # Print statistics
    # =========================================================

    print()
    print(
        f"Total words:       {len(words)}"
    )
    print(
        f"Already processed: {len(processed_words)}"
    )
    print(
        f"Remaining:         {len(words_to_process)}"
    )
    print(
        f"Processes:         {processes}"
    )
    print(
        f"Batch size:        {batch_size}"
    )

    if hf_dataset_repo:
        print(
            f"HF dataset repo:   {hf_dataset_repo}"
        )

    print()

    # =========================================================
    # Nothing left to process
    # =========================================================

    if not words_to_process:

        print(
            "All words already processed!"
        )

        return csv_path

    # =========================================================
    # Multiprocessing
    # =========================================================

    batch = []

    with Pool(
        processes=processes
    ) as pool:

        iterator = pool.imap_unordered(
            process_word,
            words_to_process,
        )

        for word, results in tqdm(
            iterator,
            total=len(words_to_process),
            desc="Processing",
        ):

            batch.append(
                (word, results)
            )

            # -------------------------------------------------
            # Batch completed
            # -------------------------------------------------

            if len(batch) >= batch_size:

                # Save locally.
                write_batch(
                    csv_path,
                    batch,
                )

                batch.clear()

                # Upload complete CSV.
                if hf_dataset_repo:

                    upload_to_huggingface(
                        hf_dataset_repo,
                        csv_path,
                    )

    # =========================================================
    # Write final partial batch
    # =========================================================

    if batch:

        write_batch(
            csv_path,
            batch,
        )

        batch.clear()

        if hf_dataset_repo:

            upload_to_huggingface(
                hf_dataset_repo,
                csv_path,
            )

    return csv_path


def main():

    args = parse_args()

    filename = obtain_video_id(
        fn_word=args.wordlist,
        outdir=args.outdir,
        processes=args.processes,
        batch_size=args.batch_size,
        hf_dataset_repo=args.hf_dataset_repo,
    )

    print()
    print(
        f"Saved video IDs to {filename}"
    )


if __name__ == "__main__":
    main()