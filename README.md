# youtube-labeled-speech-corpus
A pipeline for building large-scale ASR corpora from YouTube by leveraging manually-created subtitles.

## Pipeline Steps

### 1. Making Search Words

The first step involves creating a list of search words from the Wikipedia dump for the specified language. This is done by processing the dump file to extract relevant terms and phrases. The script downloads the compressed Wikipedia index, extracts the page titles, and cleans them by removing HTML entities (e.g., `&quot;`), file extensions (e.g., `.jpg`), numeric parentheses (e.g., `(123)`), and punctuation. The final output is a sorted, deduplicated word list.

#### Usage

To use the script for making search words, run the following command:

```bash
python -m scripts.make_search_words <language_code> --outdir <output_directory>
```

Replace `<language_code>` with the desired language code (e.g., `en` for English) and `<output_directory>` with the directory where you want to save the output files.
