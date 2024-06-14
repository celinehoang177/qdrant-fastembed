import numpy as np
import pytest

from fastembed import SparseTextEmbedding


@pytest.mark.parametrize(
    "model_name", ["Qdrant/bm42-all-minilm-l6-v2-attentions", "Qdrant/bm25"]
)
def test_attention_embeddings(model_name):
    model = SparseTextEmbedding(model_name=model_name)

    output = list(
        model.query_embed(
            [
                "I must not fear. Fear is the mind-killer.",
            ]
        )
    )

    assert len(output) == 1

    for result in output:
        assert len(result.indices) == len(result.values)
        assert np.allclose(result.values, np.ones(len(result.values)))

    quotes = [
        "I must not fear. Fear is the mind-killer.",
        "All animals are equal, but some animals are more equal than others.",
        "It was a pleasure to burn.",
        "The sky above the port was the color of television, tuned to a dead channel.",
        "In the beginning, the universe was created."
        " This has made a lot of people very angry and been widely regarded as a bad move.",
        "It's a truth universally acknowledged that a zombie in possession of brains must be in want of more brains.",
        "War is peace. Freedom is slavery. Ignorance is strength.",
        "We're not in Infinity; we're in the suburbs.",
        "I was a thousand times more evil than thou!",
        "History is merely a list of surprises... It can only prepare us to be surprised yet again.",
    ]

    output = list(model.embed(quotes))

    assert len(output) == len(quotes)

    for result in output:
        assert len(result.indices) == len(result.values)
        assert len(result.indices) > 0

    # Test support for unknown languages
    output = list(
        model.query_embed(
            [
                "привет мир!",
            ]
        )
    )

    assert len(output) == 1

    for result in output:
        assert len(result.indices) == len(result.values)
        assert len(result.indices) == 2


@pytest.mark.parametrize(
    "model_name", ["Qdrant/bm42-all-minilm-l6-v2-attentions", "Qdrant/bm25"]
)
def test_parallel_processing(model_name):
    model = SparseTextEmbedding(model_name=model_name)

    docs = ["hello world", "attention embedding"] * 100
    embeddings = list(model.embed(docs, batch_size=10, parallel=2))

    embeddings_2 = list(model.embed(docs, batch_size=10, parallel=None))

    embeddings_3 = list(model.embed(docs, batch_size=10, parallel=0))

    assert len(embeddings) == len(docs)

    for emb_1, emb_2, emb_3 in zip(embeddings, embeddings_2, embeddings_3):
        assert np.allclose(emb_1.indices, emb_2.indices)
        assert np.allclose(emb_1.indices, emb_3.indices)
        assert np.allclose(emb_1.values, emb_2.values)
        assert np.allclose(emb_1.values, emb_3.values)
