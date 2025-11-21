# summarizer.py
import re
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
from typing import Literal, List

language = Literal["en", "ne"]


class SummarizerService:
    """
    Extractive summarizer based on sentence embeddings + K-Means clustering.
    Supports English (en) and Nepali (ne).
    """

    # Pretrained model mapping
    _MODELS = {
        "en": "sentence-transformers/all-MiniLM-L6-v2",
        "ne": "Sakonii/distilbert-base-nepali",
    }

    def __init__(self):
        self.models = {}
        self.tokenizers = {}

    def _load_model(self, lang: language):
        if lang not in self.models:
            model_name = self._MODELS[lang]
            print(f"Loading model for {lang.upper()}: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.eval()  # Set to evaluation mode

            self.tokenizers[lang] = tokenizer
            self.models[lang] = model
        return self.models[lang], self.tokenizers[lang]

    @staticmethod
    def _split_into_sentences(text: str, lang: language) -> List[str]:
        """
        Simple sentence splitter using punctuation.
        """
        if lang == "ne":
            sentences = re.split(r'[।!?]+', text)
        else:
            sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

    @staticmethod
    def _find_optimal_k(embeddings: np.ndarray, max_k: int = 10) -> int:
        """
        Elbow method to choose number of clusters based on embeddings.
        """
        if len(embeddings) <= 2:
            return 1

        max_k = min(max_k, len(embeddings))
        sse = []
        K = range(2, max_k + 1)

        for k in K:
            kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
            kmeans.fit(embeddings)
            sse.append(kmeans.inertia_)

        diffs = np.diff(sse)
        if len(diffs) == 0:
            return 1
        optimal_idx = np.argmax(diffs) + 2
        return min(optimal_idx, max_k)

    def _get_sentence_embeddings(self, sentences: List[str], model, tokenizer, device) -> np.ndarray:
        embeddings = []
        for sent in sentences:
            inputs = tokenizer(
                sent,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)
            # CLS token embedding
            emb = outputs.last_hidden_state[:, 0, :].cpu().numpy().squeeze()
            embeddings.append(emb)
        return np.vstack(embeddings)

    def summarize(self, text: str, language: language = "en", ratio: float = 0.2) -> str:
        """
        Generate extractive summary.

        Args:
            text (str): Input text.
            language (en|ne): Language code.
            ratio (0.0–1.0): Proportion of sentences to keep (default 0.2).

        Returns:
            str: Extractive summary.
        """
        if not text or not text.strip():
            return ""

        model, tokenizer = self._load_model(language)
        sentences = self._split_into_sentences(text, language)
        if not sentences:
            return ""

        device = next(model.parameters()).device
        embeddings = self._get_sentence_embeddings(sentences, model, tokenizer, device)

        n_clusters = max(1, int(len(sentences) * ratio))
        n_clusters = self._find_optimal_k(embeddings, max_k=n_clusters * 2)
        n_clusters = min(n_clusters, len(sentences))

        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        kmeans.fit(embeddings)

        # Select closest sentence to each cluster centroid
        summary_indices = []
        for i in range(n_clusters):
            cluster_indices = np.where(kmeans.labels_ == i)[0]
            centroid = kmeans.cluster_centers_[i]
            closest_idx = min(cluster_indices, key=lambda idx: np.linalg.norm(embeddings[idx] - centroid))
            summary_indices.append(closest_idx)

        # Sort sentences in original order
        summary_sentences = [sentences[i] for i in sorted(summary_indices)]
        joiner = "। " if language == "ne" else ". "
        end_punct = "।" if language == "ne" else "."

        return joiner.join(summary_sentences) + end_punct
