from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Type, Union

import numpy as np

from fastembed.image.image_embedding_base import ImageEmbeddingBase
from fastembed.image.onnx_embedding import OnnxImageEmbedding


class ImageEmbedding(ImageEmbeddingBase):
    EMBEDDINGS_REGISTRY: List[Type[ImageEmbeddingBase]] = [OnnxImageEmbedding]

    @classmethod
    def list_supported_models(cls) -> List[Dict[str, Any]]:
        """
        Lists the supported models.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing the model information.

            Example:
                ```
                [
                    {
                        "model": "canavar/clip-ViT-B-32-multilingual-v1-ONNX",
                        "dim": 512,
                        "description": "Multimodal (text, image) model",
                        "size_in_GB": 0.5,
                        "sources": {
                            "hf": "canavar/clip-ViT-B-32-multilingual-v1-ONNX",
                        }
                    }
                ]
                ```
        """
        result = []
        for embedding in cls.EMBEDDINGS_REGISTRY:
            result.extend(embedding.list_supported_models())
        return result

    def __init__(
        self,
        model_name: str,
        cache_dir: Optional[str] = None,
        threads: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(model_name, cache_dir, threads, **kwargs)

        for EMBEDDING_MODEL_TYPE in self.EMBEDDINGS_REGISTRY:
            supported_models = EMBEDDING_MODEL_TYPE.list_supported_models()
            if any(model_name.lower() == model["model"].lower() for model in supported_models):
                self.model = EMBEDDING_MODEL_TYPE(model_name, cache_dir, threads, **kwargs)
                return

        raise ValueError(
            f"Model {model_name} is not supported in TextEmbedding."
            "Please check the supported models using `TextEmbedding.list_supported_models()`"
        )

    def embed(
        self,
        images: Union[Union[str, Path], Iterable[Union[str, Path]]],
        batch_size: int = 16,
        parallel: Optional[int] = None,
        **kwargs,
    ) -> Iterable[np.ndarray]:
        """
        Encode a list of documents into list of embeddings.
        We use mean pooling with attention so that the model can handle variable-length inputs.

        Args:
            images: Iterator of image paths or single image path to embed
            batch_size: Batch size for encoding -- higher values will use more memory, but be faster
            parallel:
                If > 1, data-parallel encoding will be used, recommended for offline encoding of large datasets.
                If 0, use all available cores.
                If None, don't use data-parallel processing, use default onnxruntime threading instead.

        Returns:
            List of embeddings, one per document
        """
        yield from self.model.embed(images, batch_size, parallel, **kwargs)