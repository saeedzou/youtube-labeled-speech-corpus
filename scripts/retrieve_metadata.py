import os
import sys
import time
import random
import shutil
import logging
import argparse
import functools
import multiprocessing as mp

import yt_dlp
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

try:
    from huggingface_hub import HfApi, create_repo
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


# ----------------------------------------------------------------------------
# Error classification (mirrors downloader.py so both scripts agree on which
# videos are permanently unavailable and should never be retried).
# ----------------------------------------------------------------------------
BOT_CHECK_MARKERS = ("Sign in to confirm you’re not a bot",)
AGE_RESTRICTED_MARKERS = ('Sign in to confirm your age', 'This video is age-restricted')

UNAVAILABLE_MARKERS = (
    'Video unavailable',
    'This video is not available',
    'This video is unavailable',
    'This video is no longer available',
    'has been removed by the uploader',
    'account associated with this video has been terminated',
    'is not available in your country',
    'Private video',
    'video is private',
    'This video is private',
    'This video has been removed',
    'Join this channel',
    'This video is available to this',
)
COPYRIGHT_MARKERS = (
    'copyright claim',
    'copyright grounds',
    'It was removed following a copyright removal',
    'This video was removed due to a counterfeit claim',
    'It was blocked due to the claimed content',
)
GEO_BLOCKED_MARKERS = (
    'not made this video available in your country',
)
AUTH_REQUIRED_MARKERS = ('Please sign in',)

# Videos in these categories will never succeed on retry, so once seen they
# get marked 'unavailable' here and downloader.py's should_retry_video()
# (which also treats status == 'unavailable' as terminal) will skip them.
PERMANENT_ERROR_TYPES = {'age_restricted', 'unavailable', 'copyright', 'geo_blocked', 'auth_required'}


def classify_error(msg):
    """Map a raw yt-dlp/error string to a coarse error_type."""
    if not msg:
        return 'other'
    if any(marker in msg for marker in BOT_CHECK_MARKERS):
        return 'bot_check'
    if any(marker in msg for marker in AGE_RESTRICTED_MARKERS):
        return 'age_restricted'
    if any(marker in msg for marker in COPYRIGHT_MARKERS):
        return 'copyright'
    if any(marker in msg for marker in GEO_BLOCKED_MARKERS):
        return 'geo_blocked'
    if any(marker in msg for marker in UNAVAILABLE_MARKERS):
        return 'unavailable'
    if any(marker in msg for marker in AUTH_REQUIRED_MARKERS):
        return 'auth_required'
    return 'other'


class _CapturingLogger:
    """yt-dlp logger passed via ydl_opts.

    With ignoreerrors=True, yt-dlp routes extractor failures only to the
    logger instead of raising. We stash the last error line here so the
    caller can classify it (bot-check, age-restricted, unavailable, ...)
    instead of a generic failure string.
    """

    def __init__(self, stop_event):
        self.stop_event = stop_event
        self.last_error = None

    def debug(self, msg):
        pass

    def warning(self, msg):
        if 'n challenge solving failed' in msg or 'Signature solving failed' in msg:
            return  # irrelevant: we don't use format/stream URLs, metadata-only extraction
        logging.warning(msg)

    def error(self, msg):
        logging.error(msg)
        # With verbose=True, yt-dlp calls logger.error() twice per failure:
        # once with the real message and once more with a raw traceback
        # dump. Keep the first real message so classify_error() has the
        # marker text it needs.
        if self.last_error is not None and _looks_like_traceback(msg):
            return
        self.last_error = msg
        if any(marker in msg for marker in BOT_CHECK_MARKERS):
            self.stop_event.set()


def _looks_like_traceback(msg):
    stripped = msg.lstrip()
    return stripped.startswith('File "') or stripped.startswith('Traceback')


REQUIRED_FIELDS = ['title', 'channel_id', 'duration', 'categories', 'language']


def build_ydl_opts(logger):
    """Same extraction settings as downloader.py's build_ydl_opts, minus the
    parts that only matter when actually downloading media (format,
    outtmpl, postprocessors, download_ranges, ...). We only need metadata."""
    return {
        'skip_download': True,
        'cookies': 'cookies.txt',
        'noplaylist': True,
        'ignoreerrors': True,
        'max_sleep_interval': 0.2,
        'verbose': True,
        'quiet': True,
        'extractor_args': {
                'youtube': {'player_client': ['default', 'tv_downgraded', 'web_embedded']}
            },
        'logger': logger,
    }


def get_video_info(video_id, stop_event):
    if stop_event.is_set():
        # Another worker already hit the bot-check; don't burn more requests.
        return {'video_id': video_id, 'status': 'skipped', 'error_type': None, 'error': None}

    video_url = f"https://www.youtube.com/watch?v={video_id}"
    capturing_logger = _CapturingLogger(stop_event)
    ydl_opts = build_ydl_opts(capturing_logger)

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)

        if stop_event.is_set():
            return {
                'video_id': video_id,
                'status': 'failed',
                'error_type': 'bot_check',
                'error': capturing_logger.last_error,
            }

        if not info:
            # ignoreerrors=True means a failed extraction returns None
            # instead of raising -- the real reason lives in the logger.
            error_text = capturing_logger.last_error or 'yt-dlp returned no info'
            error_type = classify_error(error_text)
            status = 'unavailable' if error_type in PERMANENT_ERROR_TYPES else 'failed'
            logging.warning(f'[{video_id}] no info returned ({error_type}): {error_text}')
            return {
                'video_id': video_id,
                'status': status,
                'error_type': error_type,
                'error': error_text,
            }

        required_info = {field: info.get(field) for field in REQUIRED_FIELDS}
        subtitles = list(info.get('subtitles', {}).keys())
        return {
            **required_info,
            'subtitles': subtitles,
            'video_id': video_id,
            'status': 'downloaded',
            'error_type': None,
            'error': None,
        }
    except Exception as e:
        error_text = str(e)
        error_type = classify_error(error_text)
        if error_type == 'bot_check':
            stop_event.set()
        logging.warning(f'[{video_id}] failed ({error_type}): {error_text}')
        status = 'unavailable' if error_type in PERMANENT_ERROR_TYPES else 'failed'
        return {
            'video_id': video_id,
            'status': status,
            'error_type': error_type,
            'error': error_text,
        }


# ----------------------------------------------------------------------------
# Hugging Face Hub sync (called from the main process only)
# ----------------------------------------------------------------------------
def push_output_csv(api, output_csv, repo_id):
    try:
        api.upload_file(
            path_or_fileobj=output_csv,
            path_in_repo=os.path.basename(output_csv),
            repo_id=repo_id,
            repo_type='dataset',
            commit_message=f'Update metadata ({time.strftime("%Y-%m-%d %H:%M:%S")})',
        )
        logging.info(f'Pushed {output_csv} to {repo_id}')
    except Exception as exc:
        logging.warning(f'Could not push {output_csv} to {repo_id}: {exc}')


def try_download_from_hf(api, output_csv, repo_id):
    """Pull the existing progress file from the HF dataset repo, if any,
    so a fresh machine/container can resume instead of starting over.
    Without this, output_csv only ever reflects local disk, which is
    empty on every new machine even though the repo has real progress."""
    try:
        from huggingface_hub import hf_hub_download
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=os.path.basename(output_csv),
            repo_type='dataset',
            token=api.token,
        )
        output_dir = os.path.dirname(output_csv)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        shutil.copy(downloaded_path, output_csv)
        logging.info(f'Downloaded existing progress from {repo_id} to {output_csv}')
        return True
    except Exception as exc:
        logging.info(f'No existing file to resume from on {repo_id} ({exc})')
        return False


def main():
    parser = argparse.ArgumentParser(description='Retrieve video information from YouTube.')
    parser.add_argument('--input_csv', type=str, required=True, help='Path to the input CSV file containing video IDs.')
    parser.add_argument('--output_csv', type=str, required=True, help='Path to the output CSV file to save video information.')
    parser.add_argument('--save_frequency', type=int, default=100, help='How often to save the results to the output CSV.')
    parser.add_argument('--num_workers', type=int, default=mp.cpu_count(), help='Number of worker processes to use.')
    parser.add_argument('--max_hours', type=float, default=11, help='Maximum number of hours to run before stopping.')
    parser.add_argument('--max_attempts', type=int, default=5, help='Skip a video after this many non-permanent failures.')
    parser.add_argument('--hf_repo_id', type=str, default=None, help='HF dataset repo to push output_csv to (progress storage).')
    parser.add_argument('--no_push', action='store_true', help='disable Hugging Face Hub pushes (results are still saved locally)')

    args = parser.parse_args()
    start_time = time.time()
    max_seconds = args.max_hours * 3600  # 11 hours by default

    # ------------------------------------------------------------------
    # HF Hub setup (moved up so we can pull existing progress *before*
    # deciding whether output_csv "exists" -- a fresh machine has no
    # local file but the repo may already hold real progress).
    # ------------------------------------------------------------------
    hf_token = os.environ.get('HF_TOKEN')
    push_enabled = (not args.no_push) and HF_AVAILABLE and bool(hf_token) and bool(args.hf_repo_id)
    if args.no_push:
        logging.info('--no_push set: results will only be saved locally.')
    elif not args.hf_repo_id:
        logging.info('--hf_repo_id not set: results will only be saved locally.')
    elif not HF_AVAILABLE:
        logging.warning('huggingface_hub is not installed; results will only be saved locally. '
                         'Run `pip install huggingface_hub` to enable pushes.')
    elif not hf_token:
        logging.warning('HF_TOKEN not set in the environment; results will only be saved locally.')

    api = HfApi(token=hf_token) if push_enabled else None
    if push_enabled:
        try:
            create_repo(args.hf_repo_id, repo_type='dataset', exist_ok=True, token=hf_token)
        except Exception as exc:
            logging.warning(f'create_repo({args.hf_repo_id}) failed (may already exist / no perms): {exc}')

    # Pull existing progress from the hub before checking local disk, so
    # a fresh machine resumes from the hub's copy instead of starting over.
    if push_enabled and not os.path.exists(args.output_csv):
        try_download_from_hf(api, args.output_csv, args.hf_repo_id)

    # Load existing data if output file exists
    if os.path.exists(args.output_csv):
        print(f"Resuming from existing file: {args.output_csv}")
        df_out = pd.read_csv(args.output_csv)
    else:
        df_out = pd.DataFrame()
    output_dir = os.path.dirname(args.output_csv)
    if output_dir:  # only make dir if a directory path is specified
        os.makedirs(output_dir, exist_ok=True)

    if not df_out.empty:
        # Rows saved by older runs of this script (before status/attempts
        # existed) only ever got written on success, so they're all 'downloaded'.
        if 'status' not in df_out.columns:
            df_out['status'] = 'downloaded'
        df_out['status'] = df_out['status'].fillna('downloaded')
        if 'attempts' not in df_out.columns:
            df_out['attempts'] = 1
        df_out['attempts'] = df_out['attempts'].fillna(1).astype(int)
        if 'error_type' not in df_out.columns:
            df_out['error_type'] = None
        if 'error' not in df_out.columns:
            df_out['error'] = None

    # Load input data
    df_in = pd.read_csv(args.input_csv)
    video_ids = df_in['video_id'].unique()

    if not df_out.empty:
        attempts_so_far = dict(zip(df_out['video_id'], df_out['attempts']))
        # 'downloaded' (succeeded) and 'unavailable' (age-restricted, copyright,
        # geo-blocked, private, removed, ...) are terminal -- retrying them
        # would just waste requests and risk tripping the bot-check again.
        permanent_mask = df_out['status'].isin(['downloaded', 'unavailable'])
        capped_mask = df_out['attempts'] >= args.max_attempts
        done_ids = set(df_out.loc[permanent_mask | capped_mask, 'video_id'])
    else:
        attempts_so_far = {}
        done_ids = set()

    videos_to_process = [vid for vid in video_ids if vid not in done_ids]
    random.shuffle(videos_to_process)

    logging.info(
        f'{len(videos_to_process)}/{len(video_ids)} videos remain retryable '
        f'({len(done_ids)} already downloaded, unavailable, or capped)'
    )

    def checkpoint(df):
        df.to_csv(args.output_csv, index=False)
        logging.info(f'Saved {len(df)} results to {args.output_csv}')
        if push_enabled:
            push_output_csv(api, args.output_csv, args.hf_repo_id)

    manager = mp.Manager()
    stop_event = manager.Event()
    worker_fn = functools.partial(get_video_info, stop_event=stop_event)

    results = []
    bot_check_hit = False
    time_limit_hit = False

    with mp.Pool(processes=args.num_workers) as pool:
        with tqdm(total=len(videos_to_process), desc="Processing videos") as pbar:
            for info in pool.imap_unordered(worker_fn, videos_to_process):
                elapsed = time.time() - start_time
                if elapsed > max_seconds:
                    print(f"\n⏰ Time limit reached ({args.max_hours}h). Saving progress and exiting...")
                    time_limit_hit = True
                    pool.terminate()
                    pool.join()
                    break

                pbar.update(1)

                if info.get('status') != 'skipped':
                    info['attempts'] = attempts_so_far.get(info['video_id'], 0) + 1
                    results.append(info)

                if info.get('error_type') == 'bot_check':
                    print("\n❌ Too many bot errors, stopping early to avoid further requests.")
                    bot_check_hit = True
                    pool.terminate()
                    pool.join()
                    break

                if len(results) >= args.save_frequency:
                    temp_df = pd.DataFrame(results)
                    df_out = pd.concat([df_out, temp_df], ignore_index=True)
                    df_out = df_out.drop_duplicates(subset='video_id', keep='last')
                    checkpoint(df_out)
                    results = []

    # Save any remaining results
    if results:
        temp_df = pd.DataFrame(results)
        df_out = pd.concat([df_out, temp_df], ignore_index=True)
        df_out = df_out.drop_duplicates(subset='video_id', keep='last')
        checkpoint(df_out)

    if bot_check_hit:
        print('*' * 15)
        print('* Stopped: YouTube bot-check triggered *')
        print('*' * 15)
        sys.exit(2)

    if time_limit_hit:
        print('*' * 15)
        print('* Stopped: max runtime reached *')
        print('*' * 15)
        sys.exit(3)

    print("Processing complete.")


if __name__ == '__main__':
    main()