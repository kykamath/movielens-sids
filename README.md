# MovieLens Semantic ID (SID) Generation

This project implements a Residual Quantized Variational Autoencoder (RQ-VAE) to generate discrete, interpretable Semantic IDs (SIDs) for movie embeddings derived from the MovieLens-32M dataset. The goal is to transform high-dimensional, continuous movie embeddings into a sequence of discrete tokens, making them more manageable and potentially more interpretable for downstream tasks like recommendation systems or content-based retrieval.

## Project Structure

*   `train.py`: The main script for training the RQ-VAE model using PyTorch Lightning. It handles data loading, model initialization, training loop, and uploading the best model to the Hugging Face Hub.
*   `rqvae_lightning.py`: Contains the PyTorch Lightning module (`RQVAE`) that wraps the `ResidualQuantizer` and the `MovieEmbeddingDataModule` for data preparation and loading. It also includes an `EmbeddingGenerator` for creating movie embeddings.
*   `residual_quantized_vae.py`: Defines the core `ResidualQuantizer` model architecture, which performs the quantization of continuous embeddings into discrete codes.
*   `models.py`: Defines data structures (like the `Movie` dataclass), constants for Hugging Face Hub repository IDs, and the name of the embedding model used.
*   `create_final_dataset.py`: A utility script to generate the final dataset. It loads the trained RQ-VAE model, generates embeddings for the MovieLens data, quantizes them into SIDs, and then uploads the enriched dataset (with embeddings and SIDs) to the Hugging Face Hub.
*   `visualize_sids.py`: A script to visualize the generated Semantic IDs using UMAP dimensionality reduction and Plotly for interactive plots. It helps in understanding the clustering and distribution of SIDs.
*   `requirements.txt`: Lists all Python dependencies required to run the project.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/movielens-sids.git
    cd movielens-sids
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3.  **Hugging Face Hub Token (Optional but Recommended):**
    To upload models and datasets to the Hugging Face Hub, you'll need an authentication token.
    *   Go to [Hugging Face Settings](https://huggingface.co/settings/tokens) and create a new token with "write" access.
    *   Create a `.env` file in the project root and add your token:
        ```
        HUGGING_FACE_HUB_TOKEN="hf_YOUR_TOKEN_HERE"
        ```
    *   Alternatively, you can log in via the CLI: `huggingface-cli login`

## Usage

### 1. Train the RQ-VAE Model

Run the training script. The best model (based on validation loss) will be saved locally and optionally uploaded to the Hugging Face Hub.

```bash
python train.py
```

The trained model will be pushed to `krishnakamath/rq-vae-movielens` on the Hugging Face Hub.

### 2. Create the Final Dataset with Semantic IDs

After training, use this script to generate embeddings and SIDs for the entire MovieLens dataset and push the enriched dataset to the Hugging Face Hub.

```bash
python create_final_dataset.py
```

The final dataset will be pushed to `krishnakamath/movielens-32m-movies-enriched-with-SIDs` on the Hugging Face Hub.

### 3. Visualize Semantic IDs

Generate an interactive UMAP visualization of the movie embeddings, colored by their first Semantic ID token. This helps in exploring the semantic clusters formed by the RQ-VAE.

```bash
python visualize_sids.py
```

This will generate an `semantic_id_visualization.html` file in your project directory. Open it in a web browser to view the interactive plot.

## Hugging Face Resources

*   **RQ-VAE Model:** [krishnakamath/rq-vae-movielens](https://huggingface.co/krishnakamath/rq-vae-movielens)
*   **Final Dataset with SIDs:** [krishnakamath/movielens-32m-movies-enriched-with-SIDs](https://huggingface.co/datasets/krishnakamath/movielens-32m-movies-enriched-with-SIDs)
*   **Enriched Source Dataset:** [krishnakamath/movielens-32m-movies-enriched](https://huggingface.co/datasets/krishnakamath/movielens-32m-movies-enriched)

## Dependencies

See `requirements.txt` for a full list of dependencies. Key libraries include:
*   `pytorch-lightning`
*   `transformers`
*   `datasets`
*   `sentence-transformers`
*   `umap-learn`
*   `plotly`
*   `pandas`

## Citation

### Original Dataset
Please cite the original MovieLens dataset if you use this data in your research:

F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19. https://doi.org/10.1145/2827872

## Acknowledgement
The Python scripts used to generate and process this dataset were developed with the assistance of Google's Gemini.
