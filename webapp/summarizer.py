from transformers import pipeline
import os
import logging

class SummarizerService:
    def __init__(self, model_name: str = "facebook/bart-large-cnn", cache_dir: str = "./model_cache", device: int = -1):
        """
        Initialize the summarizer pipeline with a local cache directory.

        Args:
            model_name (str): Hugging Face model name for summarization.
            cache_dir (str): Directory to store the cached model.
            device (int): Device index (-1 for CPU, 0 or higher for GPU).
        """
        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        try:
            self.logger.info(f"Loading model {model_name} from cache directory {cache_dir}")
            self.summarizer = pipeline(
                "summarization",
                model=model_name,
                device=device,
                cache_dir=cache_dir
            )
            self.logger.info(f"Successfully loaded model {model_name}")
        except Exception as e:
            self.logger.error(f"Failed to load model {model_name}: {str(e)}")
            raise RuntimeError(f"Model initialization failed: {str(e)}")

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
            self.logger.info("Starting summarization process")
            summary_list = self.summarizer(
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=do_sample
            )
            summary = summary_list[0]['summary_text'] if summary_list else ""
            self.logger.info("Summarization completed successfully")
            return summary
        except Exception as e:
            self.logger.error(f"Summarization failed: {str(e)}")
            return ""