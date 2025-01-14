from embedding_processor import EmbeddingProcessor as ep

embedding_processor = ep()

texts = ['Computer Science', 'Data Science']

embeddings = embedding_processor.generate_embedding(texts)

similarity_score = embedding_processor.calculate_cosine_similarity(embeddings[0], embeddings[1])

print(f"Similarity score between: {similarity_score}")
