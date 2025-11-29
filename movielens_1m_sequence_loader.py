"""
Python module to load and process the MovieLens-1M dataset into user interaction sequences.

This module provides a function to generate a list of sequences, where each sequence
is a list of movie IDs that a single user has interacted with, sorted chronologically.

Example:
    >>> from movielens_1m_sequence_loader import get_movielens_1m_sequences
    >>> user_sequences = get_movielens_1m_sequences()
    >>> print(f"Loaded {len(user_sequences)} user sequences.")
    >>> print(f"Example sequence for one user: {user_sequences[0]}")
"""

import pandas as pd
from datasets import load_dataset
from typing import List, Optional

def get_movielens_1m_sequences(
    min_sequence_length: int = 5,
    cache_dir: Optional[str] = None
) -> List[List[int]]:
    """
    Loads the reczoo/Movielens1M_m1 dataset and processes it into a list of user
    interaction sequences.

    Args:
        min_sequence_length (int): The minimum number of interactions a user must have
                                   to be included in the final dataset. Defaults to 5.
        cache_dir (Optional[str]): The directory to cache the downloaded dataset.
                                   If None, uses the default Hugging Face cache directory.

    Returns:
        List[List[int]]: A list of lists, where each inner list contains the
                         chronologically sorted movie IDs for a single user.
    """
    print("Loading ratings data from reczoo/Movielens1M_m1...")
    try:
        # Load only the 'ratings' part of the dataset
        ratings_ds = load_dataset('reczoo/Movielens1M_m1', 'ratings', cache_dir=cache_dir, split='train')
    except Exception as e:
        print(f"Failed to load dataset. Error: {e}")
        return []

    ratings_df = ratings_ds.to_pandas()
    
    # Rename columns for clarity
    ratings_df.rename(columns={'UserID': 'user_id', 'MovieID': 'movie_id', 'Timestamp': 'timestamp'}, inplace=True)

    print("Processing data into user sequences...")
    
    # --- Critical Step: Sort by user and timestamp to ensure chronological order ---
    print("Sorting ratings by user and timestamp...")
    ratings_df.sort_values(by=['user_id', 'timestamp'], inplace=True)
    
    # --- Group by user and aggregate movie IDs into a list ---
    print("Grouping interactions by user...")
    user_sequences = ratings_df.groupby('user_id')['movie_id'].apply(list)
    
    # --- Filter out users with short sequences ---
    if min_sequence_length > 0:
        print(f"Filtering for users with at least {min_sequence_length} interactions...")
        user_sequences = user_sequences[user_sequences.apply(len) >= min_sequence_length]

    # Convert the pandas Series of lists into a final list of lists
    final_sequences = user_sequences.tolist()
    
    print(f"✅ Processing complete. Found {len(final_sequences):,} user sequences.")
    
    return final_sequences

# --- Example Usage ---
if __name__ == '__main__':
    # Load the entire dataset (which typically loads train, validation, and test splits)
    dataset = load_dataset("reczoo/Movielens1M_m1")

    # Or load a specific split, like the training data
    train_data = load_dataset("reczoo/Movielens1M_m1", split="train")

    # Print the structure
    print(dataset)
    print(train_data.column_names)

    # ratings_ds = load_dataset('reczoo/Movielens1M_m1')
    # ratings_ds
    # print(len(ratings_ds['train'][1]))
    # print(len(ratings_ds['test'][1]))
    # print(len(ratings_ds['validation'][1]))
    # # 1. Get the user interaction sequences
    # # We use a small minimum length for demonstration purposes
    # sequences = get_movielens_1m_sequences(min_sequence_length=10)
    #
    # if sequences:
    #     # 2. Print summary information
    #     print(f"\nSuccessfully loaded {len(sequences)} user sequences.")
    #
    #     # 3. Inspect the first few sequences
    #     print("\n--- Example User Sequences ---")
    #     for i, seq in enumerate(sequences[:5]):
    #         print(f"User {i+1} Sequence (first 15 items): {seq[:15]} (Total length: {len(seq)})")
    #
    #     # 4. Verify the data structure
    #     print(f"\nData type of returned object: {type(sequences)}")
    #     if sequences:
    #         print(f"Data type of a single sequence: {type(sequences[0])}")
    #         if sequences[0]:
    #             print(f"Data type of an item in a sequence: {type(sequences[0][0])}")
