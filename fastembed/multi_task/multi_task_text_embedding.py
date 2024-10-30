# fastembed/multi_task/multi_task_text_embedding.py

from typing import Any, Dict, Iterable, List, Optional, Sequence, Type, Union

import numpy as np

from fastembed.common import OnnxProvider
from fastembed.multi_task.multi_task_embedding_base import MultiTaskTextEmbeddingBase


class MultiTaskTextEmbedding(MultiTaskTextEmbeddingBase):
    EMBEDDINGS_REGISTRY: List[Type[MultiTaskTextEmbeddingBase]] = []

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
                        "model": "jinaai/jina-embeddings-v3",
                        "dim": [32, 64, 128, 256, 512, 768, 1024],
                        "description": "Multi-task, multi-lingual embedding model with Matryoshka architecture",
                        "license": "cc-by-nc-4.0",
                        "size_in_GB": 2.29,
                        "sources": {
                            "hf": "jinaai/jina-embeddings-v3",
                        },
                        "model_file": "onnx/model.onnx",
                        "additional_files": ["onnx/model.onnx_data"],
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
        providers: Optional[Sequence[OnnxProvider]] = None,
        cuda: bool = False,
        device_ids: Optional[List[int]] = None,
        lazy_load: bool = False,
        **kwargs,
    ):
        super().__init__(model_name, cache_dir, threads, **kwargs)
        for EMBEDDING_MODEL_TYPE in self.EMBEDDINGS_REGISTRY:
            supported_models = EMBEDDING_MODEL_TYPE.list_supported_models()
            if any(model_name.lower() == model["model"].lower() for model in supported_models):
                self.model = EMBEDDING_MODEL_TYPE(
                    model_name,
                    cache_dir,
                    threads=threads,
                    providers=providers,
                    cuda=cuda,
                    device_ids=device_ids,
                    lazy_load=lazy_load,
                    **kwargs,
                )
                return

        raise ValueError(
            f"Model {model_name} is not supported in MultiTaskTextEmbedding. "
            "Please check the supported models using `MultiTaskTextEmbedding.list_supported_models()`"
        )

    def task_embed(
        self,
        documents: Union[str, Iterable[str]],
        task_type: str,
        batch_size: int = 256,
        parallel: Optional[int] = None,
        **kwargs,
    ) -> Iterable[np.ndarray]:
        yield from self.model.task_embed(documents, task_type, batch_size, parallel, **kwargs)
