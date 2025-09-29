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

## Further Tips and Notes

Here are some additional tips and performance considerations to help you make the most of this pipeline.

### Performance Considerations

- **Step 2: Obtaining Video IDs**: This step can be time-consuming, running at approximately 6 iterations per second in a Google Colab session. However, it only needs to be performed once. To speed up the process, you can split the initial word list into multiple smaller files and run the `obtain_video_ids.py` script on each file in parallel.

- **Step 3: Retrieving and Filtering Subtitled Videos**: This is by far the most time-intensive part of the pipeline, with a processing speed of about 1 iteration per second (if no ASR processing is needed). To manage this, consider the following strategies:
    - **Parallel Processing**: Break the large CSV file of video IDs into several smaller segments.
    - **Multiple Sessions**: Run the `retrieve_subtitled_videos.py` script on each CSV segment using separate Google Colab sessions, preferably under different Google accounts. This will help distribute the workload and speed up the data collection significantly. Run the following code block at the start of the session to verify `yt-dlp` availability:

    ```python
    ! pip install -q yt-dlp
    from google.colab import runtime
    import yt_dlp
    def test_yt_dlp():
        try:
            ydl_opts = {
                'quiet': True,
                'extractaudio': False,
                'outtmpl': '/tmp/test_video.%(ext)s',
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info_dict = ydl.extract_info('https://www.youtube.com/watch?v=kJQP7kiw5Fk', download=False)
                return True
        except Exception as e:
            print(f"❌ yt-dlp failed: {e}")
            runtime.unassign()
    test_yt_dlp()
    ```
    - **Iterative Saving & Resuming**: The script is designed to save the output CSV file iteratively. This is a crucial feature that allows you to stop and resume the process without losing your progress. You can simply point the script to the last saved CSV using the `--checkpoint` argument.
    - **⚠️ A Note on Automation**: YouTube has implemented strong measures to detect and block automated scripts and bots. Running this pipeline from your own server may result in your IP address being banned. Google Colab is currently the most reliable option for running these scripts without getting blocked. If you discover other workarounds, feel free to contribute to this project with a pull request!

### Post-processing and Channel Crawling

Once your output CSV from Step 3 has a sufficient number of rows with `good_sub = True`, you can adopt a more targeted approach to expand your dataset:

1.  **Identify High-Quality Channels**: Filter the CSV file to find the unique `channel_id`s associated with the high-quality videos.

2.  **Manual Verification**: Manually visit these channels on YouTube. Watch a few of their subtitled videos to confirm if the channel consistently provides high-quality manual subtitles.

3.  **Crawl Entire Channels**: If a channel appears to be a reliable source, you can proceed to crawl all of its videos with manual subtitles. To do this, you can modify the pipeline or use a separate script to fetch all video IDs from that specific channel. When doing a full channel crawl, you might want to adjust the `--min_cer` and `--min_wer` thresholds in Step 3 to better suit the specific quality of that channel's subtitles.

### Creating Aligned Datasets for Speech Processing

After you have crawled the audio and text pairs, the next step is to create a properly aligned dataset suitable for speech processing tasks such as Automatic Speech Recognition (ASR) or Text-to-Speech (TTS). For this, we recommend using the **[ctc-segmentation-toolkit](https://github.com/saeedzou/ctc-segmentation-toolkit)**.

This toolkit is particularly effective for handling text from sources like YouTube subtitles, which are often unpunctuated and consist of short, incomplete sentences. The original toolkit relies on punctuation to segment text, which can lead to poorly segmented, abrupt utterances.

To overcome this, the recommended repository integrates the **Segment Any Text (SAT)** model, providing more robust and linguistically coherent text segmentation. To enable this feature, make sure to set `split_using_sat` to `true` in your `recipes/config.yaml` file. This will ensure that the audio is segmented into meaningful utterances, even in the absence of traditional punctuation.

## Acknowledgements

This project relies on several fantastic open-source tools and libraries. We would like to extend our gratitude to the developers and maintainers of:

-   **[JTubeSpeech](https://github.com/sarulab-speech/jtubespeech)** for the original idea and implementation of a similar pipeline.
-   **[NVIDIA NeMo](https://github.com/NVIDIA/NeMo)** for providing powerful ASR models and a comprehensive toolkit for conversational AI.
-   **[ParsNorm](https://github.com/saeedzou/parsnorm)** for Farsi text normalization.
-   **[yt-dlp](https://github.com/yt-dlp/yt-dlp)** for the robust and feature-rich YouTube downloader.
-   **[jiwer](https://github.com/jitsi/jiwer)** for the simple and effective WER and CER calculation.
-   **[ctc-segmentation-toolkit](https://github.com/saeedzou/ctc-segmentation-toolkit)** for the invaluable tool for aligning audio and text.


## Contributing

Contributions are welcome! If you have any suggestions, bug reports, or feature requests, please open an issue or submit a pull request. We appreciate any effort to improve the pipeline and make it more accessible to the community.

For any questions or discussions, you can reach out to saeedzou2012@gmail.com.