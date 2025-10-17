import os
import yt_dlp
import argparse
import pandas as pd
from tqdm import tqdm
import time
from multiprocessing import Pool, cpu_count


def get_video_info(video_id):
    video_url = f"https://www.youtube.com/watch?v={video_id}"

    required_fields = [
        'title',
        'channel_id',
        'duration',
        'categories',
        'language'
    ]

    ydl_opts = {
        'skip_download': True,
        'cookies': 'cookies.txt',
        'quiet': True,
        'no_warnings': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)

        if not info:
            return None

        required_info = {field: info.get(field) for field in required_fields}
        subtitles = list(info.get('subtitles', {}).keys())
        return {**required_info, 'subtitles': subtitles, 'video_id': video_id}
    except Exception as e:
        print(f"Unexpected error processing video {video_id}: {str(e)}")
        if "Sign in to confirm you’re not a bot" in str(e):
            print("❌ Too many bot errors, exiting.")
            exit(1)
        return None


def main():
    parser = argparse.ArgumentParser(description='Retrieve video information from YouTube.')
    parser.add_argument('--input_csv', type=str, required=True, help='Path to the input CSV file containing video IDs.')
    parser.add_argument('--output_csv', type=str, required=True, help='Path to the output CSV file to save video information.')
    parser.add_argument('--save_frequency', type=int, default=100, help='How often to save the results to the output CSV.')
    parser.add_argument('--num_workers', type=int, default=cpu_count(), help='Number of worker processes to use.')
    parser.add_argument('--max_hours', type=float, default=11, help='Maximum number of hours to run before stopping.')

    args = parser.parse_args()
    start_time = time.time()
    max_seconds = args.max_hours * 3600  # 11 hours by default

    # Load existing data if output file exists
    if os.path.exists(args.output_csv):
        print(f"Resuming from existing file: {args.output_csv}")
        df_out = pd.read_csv(args.output_csv)
        processed_videos = set(df_out['video_id'])
    else:
        df_out = pd.DataFrame()
        processed_videos = set()
    output_dir = os.path.dirname(args.output_csv)
    if output_dir:  # only make dir if a directory path is specified
        os.makedirs(output_dir, exist_ok=True)

    # Load input data
    df_in = pd.read_csv(args.input_csv)
    video_ids = df_in['video_id'].unique()

    videos_to_process = [vid for vid in video_ids if vid not in processed_videos]

    results = []
    with Pool(processes=args.num_workers) as pool:
        with tqdm(total=len(videos_to_process), desc="Processing videos") as pbar:
            for info in pool.imap_unordered(get_video_info, videos_to_process):
                elapsed = time.time() - start_time
                if elapsed > max_seconds:
                    print("\n⏰ Time limit reached (11 hours). Saving progress and exiting...")
                    pool.terminate()
                    pool.join()
                    break
                
                if info:
                    results.append(info)

                if len(results) >= args.save_frequency:
                    temp_df = pd.DataFrame(results)
                    df_out = pd.concat([df_out, temp_df], ignore_index=True)
                    df_out.to_csv(args.output_csv, index=False)
                    results = []
                    print(f"\nSaved {len(df_out)} results to {args.output_csv}")
                
                pbar.update(1)

    # Save any remaining results
    if results:
        temp_df = pd.DataFrame(results)
        df_out = pd.concat([df_out, temp_df], ignore_index=True)
        df_out.to_csv(args.output_csv, index=False)
        print(f"\nSaved final {len(df_out)} results to {args.output_csv}")

    print("Processing complete.")


if __name__ == '__main__':
    main()
