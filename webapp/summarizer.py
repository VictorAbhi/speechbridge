from transformers import pipeline, BartForConditionalGeneration, BartTokenizer
import os
import logging

class SummarizerService:
    def __init__(self, model_name: str = "facebook/bart-large-cnn", cache_dir: str = "./model_cache", device: int = -1):
        """
        Initialize the summarizer pipeline by loading model and tokenizer from a local cache directory.

        Args:
            model_name (str): Hugging Face model name for summarization (e.g., facebook/bart-large-cnn).
            cache_dir (str): Directory containing the cached model and tokenizer.
            device (int): Device index (-1 for CPU, 0 or higher for GPU).
        """
        # Convert cache_dir to absolute path
        cache_dir = os.path.abspath(cache_dir)
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Using cache directory: {cache_dir}")
        
        # Define model path
        model_path = os.path.join(cache_dir, model_name.replace("/", "--"))
        required_files = ['pytorch_model.bin', 'config.json', 'vocab.json', 'merges.txt', 'tokenizer_config.json']
        
        # Check if model directory exists
        self.logger.info(f"Checking model directory: {model_path}")
        if not os.path.exists(model_path):
            self.logger.error(f"Model directory {model_path} does not exist.")
            raise FileNotFoundError(
                f"Model directory {model_path} not found. Ensure the directory exists and contains all required files. "
                "You can manually download the model and tokenizer from https://huggingface.co/facebook/bart-large-cnn/tree/main "
                "and place them in {model_path}."
            )
        
        # Verify required files
        missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_path, f))]
        if missing_files:
            self.logger.error(f"Missing required files in {model_path}: {', '.join(missing_files)}")
            raise FileNotFoundError(
                f"Missing files in {model_path}: {', '.join(missing_files)}. "
                "Ensure all model and tokenizer files are present. You can manually download them from "
                "https://huggingface.co/facebook/bart-large-cnn/tree/main and place them in {model_path}."
            )
        
        try:
            # Load tokenizer
            self.logger.info(f"Loading tokenizer for {model_name} from {model_path}")
            self.tokenizer = BartTokenizer.from_pretrained(
                model_path,
                local_files_only=True
            )
            self.logger.info("Tokenizer loaded successfully")

            # Load model
            self.logger.info(f"Loading model {model_name} from {model_path}")
            self.model = BartForConditionalGeneration.from_pretrained(
                model_path,
                local_files_only=True
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

        except Exception as e:
            self.logger.error(f"Failed to load model or tokenizer: {str(e)}")
            raise RuntimeError(
                f"Model or tokenizer initialization failed: {str(e)}. "
                f"Ensure all files are in {model_path} and are not corrupted. "
                "Check https://huggingface.co/facebook/bart-large-cnn/tree/main for required files."
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