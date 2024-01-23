import functools
import json
import os
import shutil
import tarfile
import tempfile
from abc import ABC, abstractmethod
from itertools import islice
from multiprocessing import get_all_start_methods
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional, Tuple, Union

import numpy as np
import onnxruntime as ort
import requests
from tokenizers import AddedToken, Tokenizer
from tqdm import tqdm
from huggingface_hub import snapshot_download
from huggingface_hub.utils import RepositoryNotFoundError
from loguru import logger

from fastembed.parallel_processor import ParallelWorkerPool, Worker


def iter_batch(iterable: Union[Iterable, Generator], size: int) -> Iterable:
    """
    >>> list(iter_batch([1,2,3,4,5], 3))
    [[1, 2, 3], [4, 5]]
    """
    source_iter = iter(iterable)
    while source_iter:
        b = list(islice(source_iter, size))
        if len(b) == 0:
            break
        yield b


def normalize(input_array, p=2, dim=1, eps=1e-12):
    # Calculate the Lp norm along the specified dimension
    norm = np.linalg.norm(input_array, ord=p, axis=dim, keepdims=True)
    norm = np.maximum(norm, eps)  # Avoid division by zero
    normalized_array = input_array / norm
    return normalized_array


class EmbeddingModel:
    @classmethod
    def load_tokenizer(cls, model_dir: Path, max_length: int = 512) -> Tokenizer:
        config_path = model_dir / "config.json"
        if not config_path.exists():
            raise ValueError(f"Could not find config.json in {model_dir}")

        tokenizer_path = model_dir / "tokenizer.json"
        if not tokenizer_path.exists():
            raise ValueError(f"Could not find tokenizer.json in {model_dir}")

        tokenizer_config_path = model_dir / "tokenizer_config.json"
        if not tokenizer_config_path.exists():
            raise ValueError(f"Could not find tokenizer_config.json in {model_dir}")

        tokens_map_path = model_dir / "special_tokens_map.json"
        if not tokens_map_path.exists():
            raise ValueError(f"Could not find special_tokens_map.json in {model_dir}")

        with open(str(config_path)) as config_file:
            config = json.load(config_file)

        with open(str(tokenizer_config_path)) as tokenizer_config_file:
            tokenizer_config = json.load(tokenizer_config_file)

        with open(str(tokens_map_path)) as tokens_map_file:
            tokens_map = json.load(tokens_map_file)

        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        tokenizer.enable_truncation(max_length=min(tokenizer_config["model_max_length"], max_length))
        tokenizer.enable_padding(pad_id=config["pad_token_id"], pad_token=tokenizer_config["pad_token"])

        for token in tokens_map.values():
            if isinstance(token, str):
                tokenizer.add_special_tokens([token])
            elif isinstance(token, dict):
                tokenizer.add_special_tokens([AddedToken(**token)])

        return tokenizer

    def __init__(
        self,
        path: Path,
        model_name: str,
        max_length: int = 512,
        max_threads: int = None,
    ):
        self.path = path
        self.model_name = model_name
        model_path = self.path / "model.onnx"
        optimized_model_path = self.path / "model_optimized.onnx"

        # List of Execution Providers: https://onnxruntime.ai/docs/execution-providers
        onnx_providers = ["CPUExecutionProvider"]

        if not model_path.exists():
            # Rename file model_optimized.onnx to model.onnx if it exists
            if optimized_model_path.exists():
                optimized_model_path.rename(model_path)
            else:
                raise ValueError(f"Could not find model.onnx in {self.path}")

        # Hacky support for multilingual model
        self.exclude_token_type_ids = False
        if model_name == "intfloat/multilingual-e5-large":
            self.exclude_token_type_ids = True

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        if max_threads is not None:
            so.intra_op_num_threads = max_threads
            so.inter_op_num_threads = max_threads

        self.tokenizer = self.load_tokenizer(self.path, max_length=max_length)
        self.model = ort.InferenceSession(str(model_path), providers=onnx_providers, sess_options=so)

    def onnx_embed(self, documents: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        encoded = self.tokenizer.encode_batch(documents)
        input_ids = np.array([e.ids for e in encoded])
        attention_mask = np.array([e.attention_mask for e in encoded])

        onnx_input = {
            "input_ids": np.array(input_ids, dtype=np.int64),
            "attention_mask": np.array(attention_mask, dtype=np.int64),
        }

        if not self.exclude_token_type_ids:
            onnx_input["token_type_ids"] = np.array(
                [np.zeros(len(e), dtype=np.int64) for e in input_ids], dtype=np.int64
            )

        model_output = self.model.run(None, onnx_input)
        embeddings = model_output[0]
        return embeddings, attention_mask


class EmbeddingWorker(Worker):
    def __init__(
        self,
        path: Path,
        model_name: str,
        max_length: int = 512,
    ):
        self.model = EmbeddingModel(path=path, model_name=model_name, max_length=max_length, max_threads=1)

    @classmethod
    def start(cls, path: Path, model_name: str, max_length: int = 512, **kwargs: Any) -> "EmbeddingWorker":
        return cls(
            path=path,
            model_name=model_name,
            max_length=max_length,
        )

    def process(self, items: Iterable[Tuple[int, Any]]) -> Iterable[Tuple[int, Any]]:
        for idx, batch in items:
            embeddings, attn_mask = self.model.onnx_embed(batch)
            yield idx, (embeddings, attn_mask)


class Embedding(ABC):
    """
    Abstract class for embeddings.

    Inherits:
        ABC: Abstract base class

    Raises:
        NotImplementedError: Raised when you call an abstract method that has not been implemented.
        PermissionError: _description_
        ValueError: Several possible reasons: 1) targz_path does not exist or is not a file, 2) targz_path is not a .tar.gz file, 3) An error occurred while decompressing targz_path, 4) Could not find model_dir in cache_dir, 5) Could not find tokenizer.json in model_dir, 6) Could not find model.onnx in model_dir.
        NotImplementedError: _description_

    Returns:
        _type_: _description_

    Yields:
        _type_: _description_
    """

    # Internal helper decorator to maintain backward compatibility
    # by supporting a fallback to download from Google Cloud Storage (GCS)
    # if the model couldn't be downloaded from HuggingFace.
    def gcs_fallback(hf_download_method: Callable) -> Callable:
        @functools.wraps(hf_download_method)
        def wrapper(self, *args, **kwargs):
            try:
                return hf_download_method(self, *args, **kwargs)
            except (EnvironmentError, RepositoryNotFoundError, ValueError) as e:
                logger.exception(
                    f"Could not download model from HuggingFace: {e}"
                    "Falling back to download from Google Cloud Storage"
                )
                return self.retrieve_model_gcs(*args, **kwargs)

        return wrapper

    @abstractmethod
    def embed(self, texts: Iterable[str], batch_size: int = 256, parallel: int = None) -> List[np.ndarray]:
        raise NotImplementedError

    @classmethod
    def list_supported_models(cls, exclude: List[str] = []) -> List[Dict[str, Any]]:
        """Lists the supported models.

        Args:
            exclude (List[str], optional): Keys to exclude from the result. Defaults to [].

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing the model information.
        """
        models_file_path = Path(__file__).with_name("models.json")
        with open(models_file_path, "r") as file:
            models = json.load(file)

        models = [{k: v for k, v in model.items() if k not in exclude} for model in models]

        return models

    @classmethod
    def download_file_from_gcs(cls, url: str, output_path: str, show_progress: bool = True) -> str:
        """
        Downloads a file from Google Cloud Storage.

        Args:
            url (str): The URL to download the file from.
            output_path (str): The path to save the downloaded file to.
            show_progress (bool, optional): Whether to show a progress bar. Defaults to True.

        Returns:
            str: The path to the downloaded file.
        """

        if os.path.exists(output_path):
            return output_path
        response = requests.get(url, stream=True)

        # Handle HTTP errors
        if response.status_code == 403:
            raise PermissionError(
                "Authentication Error: You do not have permission to access this resource. Please check your credentials."
            )

        # Get the total size of the file
        total_size_in_bytes = int(response.headers.get("content-length", 0))

        # Warn if the total size is zero
        if total_size_in_bytes == 0:
            print(f"Warning: Content-length header is missing or zero in the response from {url}.")

        show_progress = total_size_in_bytes and show_progress

        with tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True, disable=not show_progress) as progress_bar:
            with open(output_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:  # Filter out keep-alive new chunks
                        progress_bar.update(len(chunk))
                        file.write(chunk)
        return output_path

    @classmethod
    def download_files_from_huggingface(cls, model_name: str, cache_dir: Optional[str] = None) -> str:
        """
        Downloads a model from HuggingFace Hub.
        Args:
            model_name (str): Name of the model to download.
            cache_dir (Optional[str]): The path to the cache directory.
        Raises:
            ValueError: If the model_name is not in the format <org>/<model> e.g. "jinaai/jina-embeddings-v2-small-en".
        Returns:
            Path: The path to the model directory.
        """
        models = cls.list_supported_models(exclude=["compressed_url_sources"])

        hf_sources = [item for model in models if model["model"] == model_name for item in model["hf_sources"]]

        # Check if the HF sources list is empty
        # Raise an exception causing a fallback to GCS
        if not hf_sources:
            raise ValueError(f"No HuggingFace source for {model_name}")

        for index, repo_id in enumerate(hf_sources):
            try:
                return snapshot_download(
                    repo_id=repo_id,
                    ignore_patterns=["model.safetensors", "pytorch_model.bin"],
                    cache_dir=cache_dir,
                )
            except (RepositoryNotFoundError, EnvironmentError) as e:
                logger.exception(f"Failed to download model from HF source: {repo_id}: {e} ")
                if repo_id == hf_sources[-1]:
                    raise e
                logger.info(f"Trying another source: {hf_sources[index+1]}")

    @classmethod
    def decompress_to_cache(cls, targz_path: str, cache_dir: str):
        """
        Decompresses a .tar.gz file to a cache directory.

        Args:
            targz_path (str): Path to the .tar.gz file.
            cache_dir (str): Path to the cache directory.

        Returns:
            cache_dir (str): Path to the cache directory.
        """
        # Check if targz_path exists and is a file
        if not os.path.isfile(targz_path):
            raise ValueError(f"{targz_path} does not exist or is not a file.")

        # Check if targz_path is a .tar.gz file
        if not targz_path.endswith(".tar.gz"):
            raise ValueError(f"{targz_path} is not a .tar.gz file.")

        try:
            # Open the tar.gz file
            with tarfile.open(targz_path, "r:gz") as tar:
                # Extract all files into the cache directory
                tar.extractall(path=cache_dir)
        except tarfile.TarError as e:
            # If any error occurs while opening or extracting the tar.gz file,
            # delete the cache directory (if it was created in this function)
            # and raise the error again
            if "tmp" in cache_dir:
                shutil.rmtree(cache_dir)
            raise ValueError(f"An error occurred while decompressing {targz_path}: {e}")

        return cache_dir

    def retrieve_model_gcs(self, model_name: str, cache_dir: str) -> Path:
        """
        Retrieves a model from Google Cloud Storage.

        Args:
            model_name (str): The name of the model to retrieve.
            cache_dir (str): The path to the cache directory.

        Raises:
            ValueError: If the model_name is not in the format <org>/<model> e.g. BAAI/bge-base-en.

        Returns:
            Path: The path to the model directory.
        """
        fast_model_name = f"fast-{model_name.split('/')[-1]}"

        model_dir = Path(cache_dir) / fast_model_name
        if model_dir.exists():
            return model_dir

        model_tar_gz = Path(cache_dir) / f"{fast_model_name}.tar.gz"

        models = self.list_supported_models(exclude=["hf_sources"])

        compressed_url_sources = [
            item for model in models if model["model"] == model_name for item in model["compressed_url_sources"]
        ]

        # Check if the GCS sources list is empty after falling back from HF
        # A model should always have at least one source
        if not compressed_url_sources:
            raise ValueError(f"No GCS source for {model_name}")

        for index, source in enumerate(compressed_url_sources):
            try:
                self.download_file_from_gcs(
                    source,
                    output_path=str(model_tar_gz),
                )
            except (RuntimeError, PermissionError) as e:
                logger.exception(f"Failed to download model from GCS source: {source}: {e} ")
                if source == compressed_url_sources[-1]:
                    raise e
                logger.info(f"Trying another source: {compressed_url_sources[index+1]}")

        self.decompress_to_cache(targz_path=str(model_tar_gz), cache_dir=cache_dir)
        assert model_dir.exists(), f"Could not find {model_dir} in {cache_dir}"

        model_tar_gz.unlink()

        return model_dir

    @gcs_fallback
    def retrieve_model_hf(self, model_name: str, cache_dir: str) -> Path:
        """
        Retrieves a model from HuggingFace Hub.
        Args:
            model_name (str): The name of the model to retrieve.
            cache_dir (str): The path to the cache directory.
        Returns:
            Path: The path to the model directory.
        """

        return Path(self.download_files_from_huggingface(model_name=model_name, cache_dir=cache_dir))

    @classmethod
    def assert_model_name(cls, model_name: str):
        assert "/" in model_name, "model_name must be in the format <org>/<model> e.g. BAAI/bge-base-en"

        models = cls.list_supported_models()
        model_names = [model["model"] for model in models]
        if model_name not in model_names:
            raise ValueError(
                f"{model_name} is not a supported model.\n"
                f"Try one of {', '.join(model_names)}.\n"
                f"Use the 'list_supported_models()' method to get the model information."
            )

    def passage_embed(self, texts: Iterable[str], **kwargs) -> Iterable[np.ndarray]:
        """
        Embeds a list of text passages into a list of embeddings.

        Args:
            texts (Iterable[str]): The list of texts to embed.
            **kwargs: Additional keyword argument to pass to the embed method.

        Yields:
            Iterable[np.ndarray]: The embeddings.
        """

        yield from self.embed((f"passage: {t}" for t in texts), **kwargs)

    def query_embed(self, query: str) -> Iterable[np.ndarray]:
        """
        Embeds a query

        Args:
            query (str): The query to search for.

        Returns:
            Iterable[np.ndarray]: The embeddings.
        """

        # Prepend "query: " to the query
        query = f"query: {query}"
        # Embed the query
        query_embedding = self.embed([query])
        return query_embedding


class FlagEmbedding(Embedding):
    """
    Implementation of the Flag Embedding model.

    Args:
        Embedding (_type_): _description_
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        max_length: int = 512,
        cache_dir: str = None,
        threads: int = None,
    ):
        """
        Args:
            model_name (str): The name of the model to use.
            max_length (int, optional): The maximum number of tokens. Defaults to 512. Unknown behavior for values > 512.
            cache_dir (str, optional): The path to the cache directory.
                                       Can be set using the `FASTEMBED_CACHE_PATH` env variable.
                                       Defaults to `fastembed_cache` in the system's temp directory.
            threads (int, optional): The number of threads single onnxruntime session can use. Defaults to None.

        Raises:
            ValueError: If the model_name is not in the format <org>/<model> e.g. BAAI/bge-base-en.
        """

        self.assert_model_name(model_name)

        self.model_name = model_name

        if cache_dir is None:
            default_cache_dir = os.path.join(tempfile.gettempdir(), "fastembed_cache")
            cache_dir = Path(os.getenv("FASTEMBED_CACHE_PATH", default_cache_dir))
            cache_dir.mkdir(parents=True, exist_ok=True)

        self._cache_dir = cache_dir
        self._model_dir = self.retrieve_model_hf(model_name, cache_dir)
        self._max_length = max_length

        self.model = EmbeddingModel(self._model_dir, self.model_name, max_length=max_length, max_threads=threads)

    def embed(
        self, documents: Union[str, Iterable[str]], batch_size: int = 256, parallel: int = None
    ) -> Iterable[np.ndarray]:
        """
        Encode a list of documents into list of embeddings.
        We use mean pooling with attention so that the model can handle variable-length inputs.

        Args:
            documents: Iterator of documents or single document to embed
            batch_size: Batch size for encoding -- higher values will use more memory, but be faster
            parallel:
                If > 1, data-parallel encoding will be used, recommended for offline encoding of large datasets.
                If 0, use all available cores.
                If None, don't use data-parallel processing, use default onnxruntime threading instead.

        Returns:
            List of embeddings, one per document
        """
        is_small = False

        if isinstance(documents, str):
            documents = [documents]
            is_small = True

        if isinstance(documents, list):
            if len(documents) < batch_size:
                is_small = True

        if parallel == 0:
            parallel = os.cpu_count()

        if parallel is None or is_small:
            for batch in iter_batch(documents, batch_size):
                embeddings, _ = self.model.onnx_embed(batch)
                yield from normalize(embeddings[:, 0]).astype(np.float32)
        else:
            start_method = "forkserver" if "forkserver" in get_all_start_methods() else "spawn"
            params = {
                "path": self._model_dir,
                "model_name": self.model_name,
                "max_length": self._max_length,
            }
            pool = ParallelWorkerPool(parallel, EmbeddingWorker, start_method=start_method)
            for batch in pool.ordered_map(iter_batch(documents, batch_size), **params):
                embeddings, _ = batch
                yield from normalize(embeddings[:, 0]).astype(np.float32)

    @classmethod
    def list_supported_models(
        cls, exclude: List[str] = ["compressed_url_sources", "hf_sources"]
    ) -> List[Dict[str, Any]]:
        """Lists the supported models.

        Args:
            exclude (List[str], optional): Keys to exclude from the result. Defaults to ["compressed_url_sources", "hf_sources"].

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing the model information.
        """
        # jina models are not supported by this class
        return [
            model for model in super().list_supported_models(exclude=exclude) if not model["model"].startswith("jinaai")
        ]


class DefaultEmbedding(FlagEmbedding):
    """
    Implementation of the default Flag Embedding model.

    Args:
        FlagEmbedding (_type_): _description_
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        max_length: int = 512,
        cache_dir: Optional[str] = None,
        threads: Optional[int] = None,
    ):
        super().__init__(model_name, max_length=max_length, cache_dir=cache_dir, threads=threads)


class OpenAIEmbedding(Embedding):
    def __init__(self):
        # Initialize your OpenAI model here
        # self.model = ...
        ...

    def embed(self, texts, batch_size: int = 256, parallel: int = None):
        # Use your OpenAI model to embed the texts
        # return self.model.embed(texts)
        raise NotImplementedError


class JinaEmbedding(Embedding):
    def __init__(
        self,
        model_name: str = "jinaai/jina-embeddings-v2-base-en",
        max_length: int = 512,
        cache_dir: str = None,
        threads: int = None,
    ):
        """
        Args:
            model_name (str): The name of the model to use.
            max_length (int, optional): The maximum number of tokens. Defaults to 512. Unknown behavior for values > 512.
            cache_dir (str, optional): The path to the cache directory.
                                       Can be set using the `FASTEMBED_CACHE_PATH` env variable.
                                       Defaults to `fastembed_cache` in the system's temp directory.
            threads (int, optional): The number of threads single onnxruntime session can use. Defaults to None.
        Raises:
            ValueError: If the model_name is not in the format <org>/<model> e.g. jinaai/jina-embeddings-v2-base-en.
        """
        self.assert_model_name(model_name)

        self.model_name = model_name

        if cache_dir is None:
            default_cache_dir = os.path.join(tempfile.gettempdir(), "fastembed_cache")
            cache_dir = Path(os.getenv("FASTEMBED_CACHE_PATH", default_cache_dir))
            cache_dir.mkdir(parents=True, exist_ok=True)

        self._cache_dir = cache_dir
        self._model_dir = self.retrieve_model_hf(model_name, cache_dir)
        self._max_length = max_length

        self.model = EmbeddingModel(self._model_dir, self.model_name, max_length=max_length, max_threads=threads)

    def embed(
        self, documents: Union[str, Iterable[str]], batch_size: int = 256, parallel: int = None
    ) -> Iterable[np.ndarray]:
        """
        Encode a list of documents into list of embeddings.
        We use mean pooling with attention so that the model can handle variable-length inputs.
        Args:
            documents: Iterator of documents or single document to embed
            batch_size: Batch size for encoding -- higher values will use more memory, but be faster
            parallel:
                If > 1, data-parallel encoding will be used, recommended for offline encoding of large datasets.
                If 0, use all available cores.
                If None, don't use data-parallel processing, use default onnxruntime threading instead.
        Returns:
            List of embeddings, one per document
        """
        is_small = False

        if isinstance(documents, str):
            documents = [documents]
            is_small = True

        if isinstance(documents, list):
            if len(documents) < batch_size:
                is_small = True

        if parallel == 0:
            parallel = os.cpu_count()

        if parallel is None or is_small:
            for batch in iter_batch(documents, batch_size):
                embeddings, attn_mask = self.model.onnx_embed(batch)
                yield from normalize(self.mean_pooling(embeddings, attn_mask)).astype(np.float32)
        else:
            start_method = "forkserver" if "forkserver" in get_all_start_methods() else "spawn"
            params = {
                "path": self._model_dir,
                "model_name": self.model_name,
                "max_length": self._max_length,
            }
            pool = ParallelWorkerPool(parallel, EmbeddingWorker, start_method=start_method)
            for batch in pool.ordered_map(iter_batch(documents, batch_size), **params):
                embeddings, attn_mask = batch
                yield from normalize(self.mean_pooling(embeddings, attn_mask)).astype(np.float32)

    @classmethod
    def list_supported_models(
        cls, exclude: List[str] = ["compressed_url_sources", "hf_sources"]
    ) -> List[Dict[str, Any]]:
        """Lists the supported models.

        Args:
            exclude (List[str], optional): Keys to exclude from the result. Defaults to ["compressed_url_sources", "hf_sources"].

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing the model information.
        """
        # only jina models are supported by this class
        return [
            model for model in Embedding.list_supported_models(exclude=exclude) if model["model"].startswith("jinaai")
        ]

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output
        input_mask_expanded = (np.expand_dims(attention_mask, axis=-1)).astype(float)

        sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
        mask_sum = np.clip(np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None)

        return sum_embeddings / mask_sum
