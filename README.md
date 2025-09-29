# youtube-labeled-speech-corpus
A pipeline for building large-scale ASR corpora from YouTube by leveraging manually-created subtitles.

## Pipeline Steps

### 1. Making Search Words

The first step involves creating a list of search words from the Wikipedia dump for the specified language. This is done by processing the dump file to extract relevant terms and phrases. The script downloads the compressed Wikipedia index, extracts the page titles, and cleans them by removing HTML entities (e.g., `&quot;`), file extensions (e.g., `.jpg`), numeric parentheses (e.g., `(123)`), and punctuation. The final output is a sorted, deduplicated word list.

**Usage**:
To use the script for making search words, run the following command:

```bash
python -m scripts.make_search_words <language_code> --outdir <output_directory>
```

Replace `<language_code>` with the desired language code (e.g., `en` for English) and `<output_directory>` with the directory where you want to save the output files.

### 2. Obtaining Video IDs

The second step involves using the list of search words to obtain YouTube video IDs. This is done by querying the YouTube search API with each search word and extracting the video IDs from the search results. The script processes the word list in parallel to speed up the API requests and saves the results in a CSV file. You might want to change the default `cc` parameter in the `scripts/utils.py` to `True` if you want to narrow down the search results to Creative Commons licensed videos.

**Usage**:
To use the script for obtaining video IDs, run the following command:

```bash
python -m scripts.obtain_video_ids <wordlist_file> --outdir <output_directory> --processes <num_processes>
```

### 3. Retrieving and Filtering Subtitled Videos

This script is the core of the data collection pipeline. It takes the list of video IDs from the previous step and performs the following actions for each video:

1. Downloads manual subtitles: It first checks for and downloads manually-created subtitles in the specified language.

2. Pre-filters Subtitles: If manual subtitles are found, the script performs a series of checks to quickly filter out low-quality or irrelevant subtitles before downloading the audio. These checks include:

    - Punctuation Count: It counts the number of common punctuation marks (e.g., commas, periods, question marks) in the subtitle text to ensure a certain level of linguistic quality.

    - Subtitle Duration: The script calculates the total duration of all the subtitle segments to ensure there is a sufficient amount of transcribed speech.

    - Language Verification: It checks the ratio of characters to verify that the subtitle text is primarily in the target language.

3. Downloads Audio and Generates ASR Transcript: If the subtitles pass the pre-filtering stage, the script downloads the video's audio and uses a pre-trained ASR model to generate an automatic transcript.

4. Compares Transcripts and Filters: The manually-downloaded subtitle text is then compared against the ASR-generated transcript. The script calculates the Word Error Rate (WER) and Character Error Rate (CER) to measure the quality and accuracy of the manual subtitles.

5. Saves Metadata: Videos with subtitles that meet all the defined quality criteria (i.e., sufficient punctuation, duration, correct language, low WER, and low CER) are considered to have high-quality subtitles. The script saves the metadata for these videos, including the video URL, title, channel information, and the quality metrics (WER/CER), into a CSV file.

**Usage**:

```bash
python -m scripts.retrieve_subtitled_videos \
    --lang <language_code> \
    --videoidlist <video_id_list_file> \
    --model <path_to_asr_model> \ # Path to NeMo ASR model
    --outdir <output_directory> \
    --min_wer 0.8 \
    --min_cer 0.2 \
    --min_duration 10 \
    --min_punct 5
```
