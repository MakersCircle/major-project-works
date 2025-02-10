import numpy as np

def get_probabilities(frame_index):
    """Return a random probability for the given frame index."""
    np.random.seed(frame_index)  # Ensure consistent results for the same frame
    return np.random.rand()


# sample