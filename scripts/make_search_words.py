import requests
import bz2
import argparse
import sys
import re
from scripts.utils import make_dump_url
from pathlib import Path
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(
        description="Making search words from Wikipedia",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("lang", type=str, help="language code (ja, en, ...)")
    parser.add_argument("--outdir", type=str, default="word", help="dirname to save words")
    return parser.parse_args(sys.argv[1:])

def clean_line(line, lang=None):
    # Step 1: Remove HTML entities like &quot;
    line = re.sub(r'&\w+;', '', line)
    # Step 2: Remove file extensions like .txt, .jpg, etc.
    line = re.sub(r'\.\w+', '', line)
    # Step 3: Remove integers inside parentheses like (123)
    line = re.sub(r'\(\d+\)', '', line)
    # Step 4: Remove digits
    line = re.sub(r'\d+', '', line)
    # Step 5: Remove punctuation
    line = re.sub(r'[^\w\s]', '', line)
    # Step 6: Remove all NZWJ characters with space
    line = re.sub('[\u200C\u200D\u200E\u200F]', ' ', line)
    # Step 7: Remove extra white spaces
    line = re.sub(r'\s+', ' ', line).strip()

    # Language-based filtering
    if lang == "en":
        # Remove lines that don't contain any English letters
        if not re.search(r'[A-Za-z]', line):
            return ""
    else:
        # Remove lines that are all English letters or digits (no non-English chars)
        if re.fullmatch(r'[A-Za-z0-9 ]*', line):
            return ""
    return line

def make_search_word(lang, outdir="word"):
    # download wikipedia index
    url = make_dump_url(lang)
    fn_index = Path(outdir) / "dump" / lang / Path(url).name  # xxx.txt.bz2
    fn_index.parent.mkdir(parents=True, exist_ok=True)

    if not fn_index.exists():
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))  # total file size in bytes
        chunk_size = 1024 * 1024  # 1 MB chunks

        with open(fn_index, "wb") as f, tqdm(
            total=total_size, unit='B', unit_scale=True, desc="Downloading"
        ) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    pbar.update(len(chunk))

    # obtain words
    fn_word = Path(outdir) / "word" / lang / fn_index.stem
    fn_word.parent.mkdir(parents=True, exist_ok=True)

    with bz2.open(fn_index, "rt", encoding="utf-8") as f:
        lines = f.readlines()
        words = [line.strip("\n").split(":")[-1] for line in tqdm(lines, desc="Extracting words")]

    # Clean each word using the new cleaners
    words = [clean_line(w, lang=lang) for w in tqdm(set(words), desc="Cleaning words")]
    words = [w for w in words if len(w) > 0]
    words = list(set(words))
    words.sort()
    print(f"Obtained {len(words)} unique words for {lang}.")

    with open(fn_word, "w", encoding="utf-8") as f:
        f.writelines([w + "\n" for w in words])

    return fn_word

if __name__ == "__main__":
    args = parse_args()
    filename = make_search_word(args.lang, args.outdir)
    print(f"save {args.lang.upper()} words to {filename}.")
