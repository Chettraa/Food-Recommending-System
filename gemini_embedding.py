import google.generativeai as genai
import os
import time
from typing import List, Optional
import api_keys

genai.configure(api_key=GEMINI_API_KEY)

class EmbeddingProcessor:
        def __init__(self, model: str = "models/text-embedding-004"):
                """
                Initializes the EmbeddingProcessor with the specified model.

                Parameters:
                        model (str): The model to be used for generating embeddings.
                """
                self.model = model
                self.api_key = GEMINI_API_KEY

        def generate_embedding(self, texts: List[str], max_retries: int = 3, backoff_factor: float = 1.5) -> Optional[List[List[float]]]:
                """
                Generates embeddings for a list of given texts using Google's generative AI text embedding model.

                Parameters:
                        texts (List[str]): A list of texts to generate embeddings for.
                        max_retries (int): Maximum number of retry attempts in case of connection issues.
                        backoff_factor (float): Factor by which the wait time increases with each retry.

                Returns:
                        Optional[List[List[float]]]: A list of lists, each representing the embedding vector for a text if successful, None otherwise.
                """
                time.sleep(0.5)
                try:
                        genai.configure(api_key=self.api_key)
                        # Ensure input is a valid list of strings
                        if not isinstance(texts, list) or not all(isinstance(text, str) and text.strip() for text in texts):
                                raise ValueError("Input texts must be a list of non-empty strings.")

                        # Attempt API call with retries
                        for attempt in range(max_retries):
                                try:
                                        # Call Google Generative AI API for embeddings (passing a list of texts)
                                        embeddings = []
                                        for text in texts:
                                                response = genai.embed_content(
                                                        model=self.model,
                                                        content=text
                                                )
                                                embeddings.append(response['embedding'])
                                        return embeddings

                                except (ConnectionError, RemoteDisconnected) as e:
                                        print(f"Connection issue (attempt {attempt + 1}): {e}")

                                        # Check if more retries are available
                                        if attempt < max_retries - 1:
                                                # Wait with exponential backoff
                                                time.sleep(backoff_factor * (2 ** attempt))
                                        else:
                                                print("Max retries reached. Unable to generate embeddings.")
                                                return None

                                except Exception as e:
                                        print(f"API error: {e}")
                                        return None

                except Exception as e:
                        print(f"An error occurred while generating the embeddings: {e}")
                        return None

# Example usage
if __name__ == "__main__":
        processor = EmbeddingProcessor()
        result = processor.generate_embedding(["What is the meaning of life?"])
        print(result)
