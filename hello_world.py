# --- Define Unified Vocabulary ---
import numpy as np
CODEBOOK_SIZE = 64
SID_LENGTH = 3
SID_TOKENS = CODEBOOK_SIZE * SID_LENGTH  # e.g., 64 * 3 = 192 item tokens
START_TOKEN = SID_TOKENS  # 192
END_TOKEN = SID_TOKENS + 1  # 193
PAD_TOKEN = SID_TOKENS + 2  # 194
VOCAB_SIZE = SID_TOKENS + 3  # 195 total tokens


# --- Temporary movie_to_sid dictionary ---
movie_to_sid = {
    1: [10, 20, 30],
    2: [11, 21, 31],
    3: [12, 22, 32],
    4: [13, 23, 33],
    5: [14, 24, 34],
    6: [15, 25, 35],
    7: [16, 26, 36],
    8: [17, 27, 37],
    9: [18, 28, 38],
    10: [19, 29, 39]
}


# Map SID indices to the new vocabulary range (offset by 0)
def map_sid_to_vocab(sid_list):
    tokens = []
    for layer in range(SID_LENGTH):
        # The token index for a codebook is: (layer_index * CODEBOOK_SIZE) + code_index
        token_id = layer * CODEBOOK_SIZE + sid_list[layer]
        tokens.append(token_id)
    return tokens


# --- Simulate Sequential Recommendation Data ---
def create_sequence_data(movie_to_sid, num_users=100, max_seq_len=10):
    sequences = []
    for _ in range(num_users):
        # Simulate a user's movie history (3 to 10 items)
        history_len = np.random.randint(3, max_seq_len + 1)
        movie_history = np.random.choice(list(movie_to_sid.keys()), size=history_len, replace=False)

        # Convert Movie IDs to their SID tokens
        sid_history = [map_sid_to_vocab(movie_to_sid[mid]) for mid in movie_history]

        # Create (Input, Target) pairs for next-item prediction
        for i in range(len(sid_history) - 1):
            input_sequence = [START_TOKEN] + [token for sid in sid_history[:i + 1] for token in sid]
            target_sequence = [token for sid in sid_history[i + 1] for token in sid]

            # Use the previous sequence to predict the *next* item's SID tokens
            sequences.append((input_sequence, target_sequence))

    return sequences


# Create the training sequences
train_sequences = create_sequence_data(movie_to_sid, num_users=1000)
print(f"\nGenerated {len(train_sequences)} training examples.")
print(f"Example Input Sequence (Tokens): {train_sequences[0][0]}")
print(f"Example Target SID (Tokens): {train_sequences[0][1]}")
