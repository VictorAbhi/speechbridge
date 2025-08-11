from transformers import pipeline, BartForConditionalGeneration, BartTokenizer
import logging

class SummarizerService:
    def __init__(self, model_name: str = "facebook/bart-large-cnn", device: int = -1):
        """
        Initialize the summarizer pipeline by loading model and tokenizer from the default Hugging Face cache.

        Args:
            model_name (str): Hugging Face model name for summarization (e.g., facebook/bart-large-cnn).
            device (int): Device index (-1 for CPU, 0 or higher for GPU).
        """
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initializing summarizer with model: {model_name}")

        try:
            # Load tokenizer
            self.logger.info(f"Loading tokenizer for {model_name} from default cache")
            self.tokenizer = BartTokenizer.from_pretrained(
                model_name,
                local_files_only=False  # Allow initial download if not cached
            )
            self.logger.info("Tokenizer loaded successfully")

            # Load model
            self.logger.info(f"Loading model {model_name} from default cache")
            self.model = BartForConditionalGeneration.from_pretrained(
                model_name,
                local_files_only=False  # Allow initial download if not cached
            )
            self.logger.info("Model loaded successfully")

            # Initialize pipeline
            self.summarizer = pipeline(
                "summarization",
                model=self.model,
                tokenizer=self.tokenizer,
                device=device
            )
            self.logger.info(f"Successfully initialized summarization pipeline for {model_name}")

            # Update to local_files_only for subsequent runs
            self.tokenizer = BartTokenizer.from_pretrained(model_name, local_files_only=True)
            self.model = BartForConditionalGeneration.from_pretrained(model_name, local_files_only=True)
            self.summarizer = pipeline(
                "summarization",
                model=self.model,
                tokenizer=self.tokenizer,
                device=device
            )

        except Exception as e:
            self.logger.error(f"Failed to load model or tokenizer: {str(e)}")
            raise RuntimeError(
                f"Model or tokenizer initialization failed: {str(e)}. "
                "Ensure internet connection for initial download or check cache at ~/.cache/huggingface/transformers."
            )

    def summarize(self, text: str, max_length: int = 130, min_length: int = 30, do_sample: bool = False) -> str:
        """
        Generate a summary for the given text.

        Args:
            text (str): Input text to summarize.
            max_length (int): Maximum length of the summary.
            min_length (int): Minimum length of the summary.
            do_sample (bool): Whether to use sampling; deterministic by default.

        Returns:
            str: The generated summary text.
        """
        if not text or not text.strip():
            self.logger.warning("Empty or invalid input text for summarization")
            return ""

        try:
            self.logger.info(f"Starting summarization process for text (length: {len(text)} characters)")
            summary_list = self.summarizer(
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=do_sample
            )
            summary = summary_list[0]['summary_text'] if summary_list else ""
            self.logger.info(f"Summarization completed successfully (summary length: {len(summary)} characters)")
            return summary
        except Exception as e:
            self.logger.error(f"Summarization failed: {str(e)}")
            return ""