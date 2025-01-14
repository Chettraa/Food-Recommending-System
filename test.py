from embedding_processor import EmbeddingProcessor


def test_generate_embedding():
    processor = EmbeddingProcessor()
    texts = ["Hello, world!", "This is a test."]
    embeddings = processor.generate_embedding(texts)
    assert embeddings is not None
    assert len(embeddings) == len(texts)
    assert all(len(embedding) == 512 for embedding in embeddings)