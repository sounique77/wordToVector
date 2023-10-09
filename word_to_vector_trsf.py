import os
import numpy as np
import pickle
from numpy import dot
from numpy.linalg import norm


# Load the word_to_vector dictionary (word embeddings)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

with open(CURRENT_DIR + "/word_to_vector_trsf.pkl", "rb") as pk:
    word_to_vector = pickle.load(pk)


def cosine_similarity(vec_a, vec_b):
    numerator = np.dot(vec_a, vec_b)
    denominator = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    return numerator / denominator if denominator != 0 else 0.0

def similar_words(word="tree", top_k=10):
    if word not in word_to_vector:
        return []  # Return an empty list if the word is not in the embeddings

    # Calculate cosine similarity between the word and all other words
    similarities = {w: cosine_similarity(word_to_vector[word], word_to_vector[w]) for w in word_to_vector}

    # Sort words by similarity (higher similarity means closer)
    sorted_words = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

    # Get the top k closest words
    top_k_words = [w for w, _ in sorted_words[:top_k]]
    return top_k_words

# Example usage
given_word = "plant"
top_k = 10
closest_words = similar_words(given_word, top_k)
print(f"Top {top_k} closest words to '{given_word}':")
print(closest_words)

