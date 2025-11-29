# --- Define Unified Vocabulary ---
import numpy as np
CODEBOOK_SIZE = 64
SID_LENGTH = 4
SID_TOKENS = CODEBOOK_SIZE * SID_LENGTH  # e.g., 64 * 3 = 192 item tokens
START_TOKEN = SID_TOKENS  # 192
END_TOKEN = SID_TOKENS + 1  # 193
PAD_TOKEN = SID_TOKENS + 2  # 194
VOCAB_SIZE = SID_TOKENS + 3  # 195 total tokens


# --- SID Catalog Generator ---
def generate_movie_to_sid_catalog(num_movies: int, sid_length: int, codebook_size: int) -> dict:
    """
    Generates a simulated catalog mapping movie IDs to random Semantic IDs (SIDs).

    Args:
        num_movies: The total number of unique movies in the catalog.
        sid_length: The number of tokens in each SID (e.g., number of RQ-VAE layers).
        codebook_size: The number of possible values for each token (e.g., number of embeddings per layer).

    Returns:
        A dictionary where keys are movie IDs (from 1 to num_movies) and
        values are lists of integers representing the SID.
    """
    print(f"Generating a catalog of {num_movies} movies...")
    movie_to_sid = {}
    for movie_id in range(1, num_movies + 1):
        # For each movie, generate a random SID. Each token in the SID is a random
        # integer from 0 up to the codebook_size.
        sid = np.random.randint(0, codebook_size, size=sid_length).tolist()
        movie_to_sid[movie_id] = sid
    print("✅ Catalog generation complete.")
    return movie_to_sid


# --- Map SID indices to the new vocabulary range ---
def map_sid_to_vocab(sid_list: list, sid_length: int, codebook_size: int) -> list:
    """
    Maps a list of SID tokens to their corresponding indices in the unified vocabulary.
    """
    tokens = []
    for layer in range(sid_length):
        # The token index for a codebook is: (layer_index * codebook_size) + code_index
        token_id = layer * codebook_size + sid_list[layer]
        tokens.append(token_id)
    return tokens


# --- Simulate Sequential Recommendation Data ---
def create_sequence_data(movie_to_sid: dict, num_users: int, max_seq_len: int) -> list:
    """
    Simulates user interaction data for next-item prediction training.
    """
    sequences = []
    movie_ids = list(movie_to_sid.keys())
    
    for _ in range(num_users):
        # Simulate a user's movie history (3 to max_seq_len items)
        history_len = np.random.randint(3, max_seq_len + 1)
        movie_history = np.random.choice(movie_ids, size=history_len, replace=False)

        # Convert Movie IDs to their SID tokens using the mapping function
        sid_history = [map_sid_to_vocab(movie_to_sid[mid], SID_LENGTH, CODEBOOK_SIZE) for mid in movie_history]

        # Create (Input, Target) pairs for next-item prediction
        # This simulates the task: "Given the history up to item i, predict item i+1"
        for i in range(len(sid_history) - 1):
            # Input consists of a start token followed by all SID tokens from the history so far
            input_sequence = [START_TOKEN] + [token for sid in sid_history[:i + 1] for token in sid]
            
            # Target is the list of SID tokens for the *next* item in the sequence
            target_sequence = sid_history[i + 1]

            sequences.append((input_sequence, target_sequence))

    return sequences


# --- Main Execution ---
if __name__ == "__main__":
    # 1. Generate the movie catalog
    NUM_MOVIES_IN_CATALOG = 5000
    movie_to_sid = generate_movie_to_sid_catalog(
        num_movies=NUM_MOVIES_IN_CATALOG,
        sid_length=SID_LENGTH,
        codebook_size=CODEBOOK_SIZE
    )

    # 2. Create the training sequences from the catalog
    train_sequences = create_sequence_data(
        movie_to_sid=movie_to_sid,
        num_users=10000,  # Simulate 10,000 users
        max_seq_len=20     # Max history length per user
    )
    
    print(f"\nGenerated {len(train_sequences)} training examples.")
    
    # 3. Display an example
    if train_sequences:
        example_input, example_target = train_sequences[0]
        print(f"Example Input Sequence (Tokens): {example_input}")
        print(f"Example Target SID (Tokens): {example_target}")

        # To make it clearer, let's see the original movie IDs
        # (This part is just for demonstration and not needed for training)
        first_user_history_len = (len(example_input) - 1) // SID_LENGTH
        print(f"This example was generated from a user history of {first_user_history_len} movie(s).")
        print(f"The model's task is to predict the {SID_LENGTH} tokens of the next movie.")
