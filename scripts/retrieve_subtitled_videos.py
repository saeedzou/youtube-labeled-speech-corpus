import time
import csv
import argparse
import sys
import re
import os
import yt_dlp
import subprocess
import string
import re
from pathlib import Path
from scripts.utils import make_video_url
from tqdm import tqdm

def load_audio(file_path):
    waveform, sample_rate = librosa.load(file_path, sr=16000)
    # convert to mono
    waveform = librosa.to_mono(waveform)
    # Normalize and convert to float32
    if waveform.dtype == 'int16':
        waveform = waveform.astype('float32') / 32768.0
    elif waveform.dtype == 'int32':
        waveform = waveform.astype('float32') / 2147483648.0
    elif waveform.dtype == 'uint8':
        waveform = (waveform.astype('float32') - 128) / 128.0
    else:
        # If already float32, ensure no further normalization is done
        waveform = waveform.astype('float32')

    return waveform, sample_rate

def load_model(model_path: str):
    try:
        if os.path.exists(model_path):
            model = ASRModel.restore_from(restore_path=model_path)
        else:
            model = ASRModel.from_pretrained(model_path)
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        exit(1)


def transcribe_chunk(audio_chunk, model):
    transcription = model.transcribe([audio_chunk], batch_size=1, verbose=False)
    return transcription[0].text

def transcribe_audio(file_path, model, normalizer, chunk_size=30*16000):
    waveform, _ = load_audio(file_path)
    transcriptions = []
    for start in range(0, len(waveform), chunk_size):
        end = min(len(waveform), start + chunk_size)
        if end - start < 512:
            continue
        transcription = transcribe_chunk(waveform[start:end], model)
        transcriptions.append(transcription)

    # Combine all transcriptions and normalize the final result
    final_transcription = ' '.join(transcriptions)
    final_transcription = re.sub(' +', ' ', final_transcription)
    final_transcription = normalizer.normalize(final_transcription)
    
    return final_transcription

def count_common_punctuations(text, lang):
    """Count common punctuation marks in text."""
    if lang == 'fa':
        common_punctuation_marks = r'[؟،]'
    else:
        common_punctuation_marks = r'[,?]'
    matches = re.findall(common_punctuation_marks, text)
    return len(matches)

def count_other_punctuations(text, lang):
    if lang == 'fa':
        other_punctuation_marks = r'[!؛:]'
    else:
        other_punctuation_marks = r'[.:;!]'
    matches = re.findall(other_punctuation_marks, text)
    return len(matches)

def parse_timestamp(timestamp: str) -> float:
    """Convert WebVTT or SRT timestamp to seconds."""
    # Normalize separator for milliseconds (replace ',' with '.')
    timestamp = timestamp.replace(',', '.')
    
    # Format: HH:MM:SS.mmm
    hours, minutes, seconds = timestamp.split(':')
    seconds, milliseconds = seconds.split('.')
    
    total_seconds = (
        int(hours) * 3600 +
        int(minutes) * 60 +
        int(seconds) +
        int(milliseconds) / 1000
    )
    return total_seconds


def calculate_subtitle_duration(subtitle_file: str) -> float:
    """Calculate total duration covered by subtitles (VTT or SRT)."""
    total_duration = 0.0
    try:
        with open(subtitle_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # For VTT, skip header lines
        if subtitle_file.lower().endswith(".vtt"):
            lines = lines[3:]

        for line in lines:
            if '-->' in line:
                # Extract start and end times
                start, end = line.strip().split(' --> ')
                start_time = parse_timestamp(start)
                end_time = parse_timestamp(end)
                duration = end_time - start_time
                total_duration += duration

    except Exception as e:
        print(f"❌ Error calculating subtitle duration in {subtitle_file}: {e}")
        return 0.0

    return round(total_duration, 2)

def extract_text_from_subtitle(subtitle_file):
    """Extract plain text from subtitle file (VTT or SRT), removing timings and indexes."""
    text = ""
    try:
        ext = os.path.splitext(subtitle_file)[1].lower()

        with open(subtitle_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # VTT has a header we should skip
        if ext == ".vtt":
            lines = lines[3:]

        for line in lines:
            line = line.strip()
            if not line:
                continue
            if re.match(r'^\d+$', line):  # subtitle index in SRT
                continue
            if '-->' in line:  # timestamp line
                continue
            text += line + " "

    except Exception as e:
        print(f"❌ Error reading subtitle file: {e}")
        return ""

    return text.strip()

def extract_subtitle_text(subtitle_file: str, normalizer) -> str:
    if not subtitle_file or not os.path.exists(subtitle_file):
        return None

    with open(subtitle_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    text_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Skip VTT header
        if line.startswith("WEBVTT"):
            continue
        # Skip SRT indexes (lines that are only numbers)
        if re.match(r'^\d+$', line):
            continue
        # Skip timestamp lines
        if '-->' in line:
            continue
        text_lines.append(line)

    text = " ".join(text_lines)

    # Remove text between parentheses
    text = re.sub(r'\([^)]*\)', '', text)

    # Remove text between square brackets
    text = re.sub(r'\[[^\]]*\]', '', text)

    # Remove text between asterisks
    text = re.sub(r'\*[^*]*\*', '', text)

    # Remove emails
    text = re.sub(r'\b[\w.-]+?@\w+?\.\w+?\b', '', text)

    # Remove URLs
    text = re.sub(r'\b(?:http[s]?://|www\.)\S+\b', '', text)

    # Normalize text (assuming normalizer is defined elsewhere)
    text = normalizer.normalize(text)

    return text.strip()

def download_video(video_id: str):
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    os.makedirs('videos', exist_ok=True)
    output_template = f"videos/{video_id}.%(ext)s"
    ydl_opts = {
        'format': 'bestaudio/best',         # Download best audio quality
        'outtmpl': output_template,
        'skip_download': False,             # Download the audio
        'quiet': True,
        'no_warnings': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=True)
        audio_file = ydl.prepare_filename(info).replace('.%(ext)s', info['ext'])

        return audio_file

def download_captions(video_id, lang, use_auto=True):
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    os.makedirs('subtitles', exist_ok=True)
    output_template = f"subtitles/{video_id}.%(ext)s"
    
    # yt-dlp language handling
    if lang == 'fa':
        lang_list = ['fa', 'fa-IR']
    else:
        lang_list = [lang]

    ydl_opts = {
        'outtmpl': output_template,
        'writesubtitles': not use_auto,   # Only download manual captions
        'writeautomaticsubs': use_auto,   # Only download auto captions
        'subtitleslangs': lang_list,
        'skip_download': True,
        'cookies': 'cookies.txt',
        'quiet': True,
        'no_warnings': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=True)

        subtitle_file = None

        if use_auto:
            # --- Get AUTO captions ---
            auto_caps = info.get("automatic_captions", {})
            if lang in auto_caps:
                # Prefer .srt format
                chosen = next((c for c in auto_caps[lang] if c["ext"] == "srt"), auto_caps[lang][0])
                url = chosen["url"]
                subtitle_file = f"subtitles/{video_id}.auto.{lang}.{chosen['ext']}"
                try:
                    import requests
                    r = requests.get(url)
                    r.raise_for_status()
                    with open(subtitle_file, "wb") as f:
                        f.write(r.content)
                except Exception as e:
                    subtitle_file = None
        else:
            # --- Get MANUAL captions ---
            for i in lang_list:
                potential_file = f"subtitles/{video_id}.{i}.vtt"
                if os.path.exists(potential_file):
                    subtitle_file = potential_file
                    break

        return subtitle_file, info

def check_language_ratio(text, no_english, english, max_lang_ratio, min_lang_ratio):
    english_chars = sum(1 for char in text if char in string.ascii_letters)
    total_chars = len(text)
    if total_chars == 0:
        return True  # No text to check

    ratio = english_chars / total_chars
    if no_english and ratio > max_lang_ratio:
        return False
    if english and ratio < min_lang_ratio:
        return False
    return True

def process_video(videoid, query_phrase, lang, model, normalizer, no_english, english, max_lang_ratio, min_lang_ratio, min_duration, min_wer, min_cer, min_punct, use_auto, use_asr):
    """Process a single video to get metadata, download subtitles, and analyze punctuation."""
    url = make_video_url(videoid)
    entry = {
        "videoid": videoid,
        "videourl": url,
        "language": "",
        "good_sub": "False",
        "sub": "False",
        "title": "",
        "query_phrase": query_phrase,
        "channel": "",
        "channel_id": "",
        "channel_url": "",
        "channel_follower_count": "",
        "upload_date": "",
        "duration": "",
        "view_count": "",
        "categories": [],
        "like_count": "",
        "punctuation_count": 0,
        "subtitle_duration": 0,  
        "cer": "",
        "wer": "",
    }


    try:
        # First request: Get subtitle info
        subtitle_filename, metadata = download_captions(videoid, lang, use_auto=use_auto)
        if "language" in metadata:
            entry["language"] = metadata["language"]
            if metadata["language"] != lang:
                return entry   # stop further processing
        manu_lang = list(metadata['automatic_captions'].keys())
        has_subtitle = lang in manu_lang
        entry["sub"] = str(has_subtitle)
        try:
            entry.update({
                'title': metadata.get('title', ''),
                'channel': metadata.get('channel', ''),
                'channel_id': metadata.get('channel_id', ''),
                'channel_url': metadata.get('channel_url', ''),
                'channel_follower_count': metadata.get('channel_follower_count', ''),
                'upload_date': metadata.get('upload_date', ''),
                'uploader_id': metadata.get('uploader_id', ''),
                'uploader_url': metadata.get('uploader_url', ''),
                'duration': metadata.get('duration', ''),
                'view_count': metadata.get('view_count', ''),
                'categories': metadata.get('categories', []),
                'like_count': metadata.get('like_count', '')
            })
        except Exception as e:
            print(f"❌ Error updating metadata: {e}") 

        if has_subtitle and subtitle_filename:
            print(f"❕ Downloaded subtitle for video {videoid} to {subtitle_filename}")

            # Extract text and count punctuations
            if Path(subtitle_filename).exists():
                subtitle_text = extract_text_from_subtitle(subtitle_filename)
                common_punct = count_common_punctuations(subtitle_text, lang)
                other_punct = count_other_punctuations(subtitle_text, lang)
                punct_count = common_punct + other_punct
                entry["punctuation_count"] = punct_count
                
                # Calculate total subtitle duration
                subtitle_duration = calculate_subtitle_duration(subtitle_filename)
                entry["subtitle_duration"] = round(subtitle_duration, 2)
                
                if not check_language_ratio(subtitle_text, no_english, english, max_lang_ratio, min_lang_ratio):
                    return entry

                if (entry["subtitle_duration"] > min_duration) and (common_punct > min_punct or other_punct > min_punct):
                    if use_asr:
                        print(f"❕ Downloading and processing audio for video {videoid}")
                        print(url)
                        audio_file = download_video(videoid)
                        auto_transcription = transcribe_audio(audio_file, model, normalizer)
                        
                        # Save ASR transcript to a text file
                        os.makedirs('transcripts', exist_ok=True)
                        transcript_filepath = os.path.join('transcripts', f'{videoid}.txt')
                        with open(transcript_filepath, 'w', encoding='utf-8') as f:
                            f.write(auto_transcription)
                        
                        manual_transcription = extract_subtitle_text(subtitle_filename, normalizer)
                        word_error_rate = wer(manual_transcription, auto_transcription)
                        character_error_rate = cer(manual_transcription, auto_transcription)
                        entry["wer"] = word_error_rate
                        entry["cer"] = character_error_rate
                        if word_error_rate < min_wer and character_error_rate < min_cer:
                            entry["good_sub"] = str(True)


    except subprocess.CalledProcessError as e:
        print(f"❌ Error processing video {videoid}. stdout: {e.stdout}, stderr: {e.stderr}")
    except Exception as e:
        print(f"Unexpected error processing video {videoid}: {str(e)}")
        if "Sign in to confirm you’re not a bot" in str(e):
            print("❌ Too many bot errors, exiting.")
            exit(1)

    return entry

def retrieve_subtitle_exists(lang, fn_videoid, model, normalizer, outdir="sub", wait_sec=0.2, fn_checkpoint=None, no_english=False, english=False, max_lang_ratio=0.5, min_lang_ratio=0.5, min_duration=10, min_wer=0.8, min_cer=0.2, min_punct=5, use_auto=True, use_asr=True):
    fn_sub = Path(outdir) / f"{Path(fn_videoid).stem}.csv"
    fn_sub.parent.mkdir(parents=True, exist_ok=True)

    # Load checkpoint if provided
    subtitle_exists = []
    processed_videoids = set()
    if fn_checkpoint and Path(fn_checkpoint).exists():
        with open(fn_checkpoint, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                subtitle_exists.append(row)
                processed_videoids.add(row["videoid"])

    # Load video ID list
    video_ids = []
    with open(fn_videoid, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_ids.append((row["video_id"], row['word']))

    # Define fieldnames for CSV
    fieldnames = ["videoid", 
                  "videourl", 
                  "language",
                  "title", 
                  "good_sub", 
                  "sub",
                  "wer", 
                  "cer",
                  "channel", 
                  "channel_id", 
                  "channel_url",
                  "channel_follower_count", 
                  "view_count", 
                  "like_count", 
                  "uploader_id",
                  "uploader_url",
                  "upload_date", 
                  "duration", 
                  "punctuation_count", 
                  "subtitle_duration",
                  "query_phrase",
                  "categories", 
                  ]


    # Process videos
    for videoid, query_phrase in tqdm(video_ids):
        if videoid in processed_videoids:
            continue

        entry = process_video(videoid=videoid,
                              query_phrase=query_phrase,
                              lang=lang,
                              model=model,
                              normalizer=normalizer,
                              no_english=no_english,
                              english=english,
                              max_lang_ratio=max_lang_ratio,
                              min_lang_ratio=min_lang_ratio,
                              min_duration=min_duration,
                              min_wer=min_wer,
                              min_cer=min_cer,
                              min_punct=min_punct,
                              use_auto=use_auto,
                              use_asr=use_asr)
        subtitle_exists.append(entry)

        if wait_sec > 0.01:
            time.sleep(wait_sec)

        # Write current result every 50 videos
        if len(subtitle_exists) % 50 == 0:
            with open(fn_sub, "w", newline="", encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(subtitle_exists)

    # Final write
    with open(fn_sub, "w", newline="", encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(subtitle_exists)

    return fn_sub

def main():
    """Command line execution."""
    parser = argparse.ArgumentParser(
        description="Retrieve video metadata and subtitle availability status.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--lang", type=str, required=True, help="language code (ja, en, ...)")
    parser.add_argument("--videoidlist", type=str, required=True, help="filename of video ID list")
    parser.add_argument("--model", type=str, default=None, help="Path to local .nemo model or Hugging Face model name")
    parser.add_argument("--outdir", type=str, default="output", help="dirname to save results")
    parser.add_argument("--use_auto", action='store_true', default=False, help="Whether to download automatic subtitles (default: False).")
    parser.add_argument("--use_asr", action='store_true', default=False, help="Whether to download video and pass through ASR (default: False).")
    parser.add_argument("--checkpoint", type=str, default=None, help="filename of list checkpoint (for restart retrieving)")
    parser.add_argument("--min_duration", type=float, default=10.0, help="Minimum subtitle duration in seconds.")
    parser.add_argument("--min_wer", type=float, default=0.3, help="Maximum word error rate.")
    parser.add_argument("--min_cer", type=float, default=0.2, help="Maximum character error rate.")
    parser.add_argument("--min_punct", type=int, default=0, help="Minimum common punctuation count.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--no_english", action='store_true', help="Check if text has less than max_lang_ratio of English characters")
    group.add_argument("--english", action='store_true', help="Check if text has more than min_lang_ratio of English characters")
    parser.add_argument("--max_lang_ratio", type=float, default=0.5, help="Maximum ratio of English characters allowed when --no_english is set")
    parser.add_argument("--min_lang_ratio", type=float, default=0.5, help="Minimum ratio of English characters required when --english is set")
    args = parser.parse_args()

    if not args.english and '--min_lang_ratio' in sys.argv:
        parser.error("--min_lang_ratio can only be used with --english")

    if not args.no_english and '--max_lang_ratio' in sys.argv:
        parser.error("--max_lang_ratio can only be used with --no_english")

    if args.use_asr and not args.model:
        parser.error("--model is required when --use_asr is set.")

    model = None
    normalizer = None
    if args.use_asr:
        from nemo.collections.asr.models import ASRModel
        import librosa
        from jiwer import wer, cer
        from scripts.normalizer import TextNormalizer
        model = load_model(args.model)
        normalizer = TextNormalizer(lang=args.lang)
    filename = retrieve_subtitle_exists(
        lang=args.lang,
        fn_videoid=args.videoidlist,
        model=model,
        normalizer=normalizer,
        outdir=args.outdir,
        fn_checkpoint=args.checkpoint,
        no_english=args.no_english,
        english=args.english,
        max_lang_ratio=args.max_lang_ratio,
        min_lang_ratio=args.min_lang_ratio,
        min_duration=args.min_duration,
        min_wer=args.min_wer,
        min_cer=args.min_cer,
        min_punct=args.min_punct,
        use_auto=args.use_auto,
        use_asr=args.use_asr
    )
    print(f"Saved {args.lang.upper()} subtitle info, metadata, and punctuation counts to {filename}.")


if __name__ == "__main__":
    main()
