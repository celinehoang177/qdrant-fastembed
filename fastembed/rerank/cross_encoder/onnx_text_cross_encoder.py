from typing import List, Union, Iterable, Dict, Any, Sequence, Optional


from fastembed.common import OnnxProvider
from fastembed.rerank.cross_encoder.onnx_text_model import OnnxCrossEncoderModel
from fastembed.rerank.cross_encoder.text_cross_encoder_base import TextCrossEncoderBase
from fastembed.common.utils import define_cache_dir

supported_onnx_models = [
    {
        "model": "Xenova/ms-marco-MiniLM-L-6-v2",
        "size_in_GB": 0.08,
        "sources": {
            "hf": "Xenova/ms-marco-MiniLM-L-6-v2",
        },
        "model_file": "onnx/model.onnx",
        "description": "MiniLM-L-6-v2 model optimized for re-ranking tasks.",
    },
    {
        "model": "Xenova/ms-marco-MiniLM-L-12-v2",
        "size_in_GB": 0.12,
        "sources": {
            "hf": "Xenova/ms-marco-MiniLM-L-12-v2",
        },
        "model_file": "onnx/model.onnx",
        "description": "MiniLM-L-12-v2 model optimized for re-ranking tasks.",
    },
    {
        "model": "BAAI/bge-reranker-base",
        "size_in_GB": 1.04,
        "sources": {
            "hf": "BAAI/bge-reranker-base",
        },
        "model_file": "onnx/model.onnx",
        "description": "BGE reranker base model for cross-encoder re-ranking.",
    },
]


class OnnxTextCrossEncoder(TextCrossEncoderBase, OnnxCrossEncoderModel):
    ONNX_OUTPUT_NAMES = None

    @classmethod
    def list_supported_models(cls) -> List[Dict[str, Any]]:
        return supported_onnx_models

    def __init__(
        self,
        model_name: str,
        cache_dir: Optional[str] = None,
        threads: Optional[int] = None,
        providers: Optional[Sequence[OnnxProvider]] = None,
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=threads,
            providers=providers,
            **kwargs,
        )

        model_description = self._get_model_description(model_name)
        self.cache_dir = define_cache_dir(cache_dir)
        model_dir = self.download_model(
            model_description, self.cache_dir, local_files_only=self._local_files_only
        )

        self.load_onnx_model(
            model_dir=model_dir,
            model_file=model_description["model_file"],
            threads=threads,
            providers=providers,
        )

    def rerank(
        self,
        query: str,
        documents: Union[str, Iterable[str]],
        **kwargs,
    ) -> Iterable[float]:
        """
        Reranks the provided documents based on their relevance to the query using the cross-encoder model.

        Args:
            query (str): The query string.
            documents (Union[str, Iterable[str]]): A list of documents to be ranked.

        Returns:
            Iterable[float]: A list of scores corresponding to the relevance of each document to the query.
        """
        if not documents:
            return []
        return self.onnx_embed(query, documents)