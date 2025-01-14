import time
from typing import Optional, List, Union
import numpy as np

class CalculateCosine:
    def calculate_cosine_similarity(student_major_vector: Union[np.ndarray, list], scholarship_major_vector: Union[np.ndarray, list]) -> float:
        """
        Calculate the cosine similarity between two vectors.

        Parameters:
        - student_major_vector (Union[np.ndarray, list]): The first vector (student major vector).
        - scholarship_major_vector (Union[np.ndarray, list]): The second vector (scholarship major vector).

        Returns:
        - float: The cosine similarity between the two vectors, or NaN if dimensions do not match.

        Raises:
        - ValueError: If the input vectors are empty or not of the same dimension.
        """
        # Convert lists to numpy arrays if needed
        vector_a = np.asarray(student_major_vector)
        vector_b = np.asarray(scholarship_major_vector)

        # Check if vectors are empty
        if vector_a.size == 0 or vector_b.size == 0:
            print("One of the input vectors is empty.")
            return float('nan')

        # Check if vectors have the same dimension
        if vector_a.shape != vector_b.shape:
            print("Input vectors must have the same dimensions.")
            return float('nan')  # Return NaN or some indicator of a dimension mismatch.

        # Calculate cosine similarity
        dot_product = np.dot(vector_a, vector_b)
        norm_a = np.linalg.norm(vector_a)
        norm_b = np.linalg.norm(vector_b)

        if norm_a == 0 or norm_b == 0:
            print("One of the input vectors is a zero vector.")
            return float('nan')  # Return NaN if a zero vector is detected

        cosine_similarity = dot_product / (norm_a * norm_b)
        return cosine_similarity
