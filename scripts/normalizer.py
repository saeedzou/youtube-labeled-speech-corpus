# normalizer.py

# Attempt to import language-specific dependencies and set availability flags.
try:
    from parsnorm import ParsNorm
    PARSNORM_AVAILABLE = True
except ImportError:
    PARSNORM_AVAILABLE = False

try:
    # Use the nemo_text_processing library for robust English normalization.
    from nemo_text_processing.text_normalization.normalize import normalize
    NEMO_AVAILABLE = True
except ImportError:
    NEMO_AVAILABLE = False


class TextNormalizer:
    """
    A unified text normalization class for Automatic Speech Recognition (ASR).

    This class provides a simple interface to normalize text for different languages.
    It uses ParsNorm for Farsi ('fa') and NVIDIA's NeMo toolkit for English ('en').
    
    Attributes:
        lang (str): The language code for the normalizer instance.
    """

    def __init__(self, lang: str):
        """
        Initializes the normalizer for a specified language.

        Args:
            lang (str): The language code, either 'en' for English or 'fa' for Farsi.
        
        Raises:
            ValueError: If the specified language is not supported.
            ImportError: If the required dependency for the language is not installed.
        """
        self.lang = lang.lower()

        if self.lang == 'fa':
            if not PARSNORM_AVAILABLE:
                raise ImportError(
                    "Farsi dependency not found. Please install ParsNorm with: pip install parsnorm"
                )
            self._normalizer = ParsNorm()
            self._normalize_func = self._normalizer.normalize
        
        elif self.lang == 'en':
            if not NEMO_AVAILABLE:
                raise ImportError(
                    "English dependency not found. Please install NeMo with: pip install nemo_toolkit[nlp]"
                )
            # NeMo's normalize function is used directly.
            # We wrap it in a lambda to maintain a consistent internal API.
            self._normalize_func = lambda text: normalize(text, lang='en')
        
        else:
            raise ValueError(f"Language '{lang}' is not supported. Available languages: 'en', 'fa'.")

    def normalize(self, text: str) -> str:
        """
        Normalizes the input text string.

        Args:
            text (str): The text to be normalized.

        Returns:
            str: The normalized text, converted to its spoken form.
        """
        return self._normalize_func(text)


# --- Example Usage ---
if __name__ == '__main__':
    # English Example
    # Requires: pip install nemo_toolkit[nlp]
    if NEMO_AVAILABLE:
        print("--- 🇺🇸 English Normalization Example ---")
        try:
            en_normalizer = TextNormalizer(lang='en')
            en_text = "The price is $1,234.56 at 123 Main St., NW."
            normalized_en_text = en_normalizer.normalize(en_text)
            print(f"Original:   '{en_text}'")
            print(f"Normalized: '{normalized_en_text}'")
        except Exception as e:
            print(f"An error occurred during English normalization: {e}")
    else:
        print("Skipping English example: 'nemo_toolkit[nlp]' is not installed.")

    print("\n" + "="*40 + "\n")

    # Farsi Example
    # Requires: pip install parsnorm
    if PARSNORM_AVAILABLE:
        print("--- 🇮🇷 Farsi/Persian Normalization Example ---")
        try:
            fa_normalizer = TextNormalizer(lang='fa')
            # "The flight is on 1403/05/21 at 14:30."
            fa_text = "پرواز در تاریخ ۱۴۰۳/۰۵/۲۱ ساعت ۱۴:۳۰ است."
            normalized_fa_text = fa_normalizer.normalize(fa_text)
            print(f"Original:   '{fa_text}'")
            print(f"Normalized: '{normalized_fa_text}'")
        except Exception as e:
            print(f"An error occurred during Farsi normalization: {e}")
    else:
        print("Skipping Farsi example: 'parsnorm' is not installed.")