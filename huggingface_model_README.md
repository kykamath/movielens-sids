---
tags:
- pytorch
- pytorch-lightning
- vae
- vector-quantization
- embeddings
- movielens
- recommendation-systems
- semantic-search
---

# RQ-VAE Movie Embedding Quantizer

This model is a Residual Quantized Variational Autoencoder (RQ-VAE) trained to quantize high-dimensional movie embeddings into discrete Semantic IDs (SIDs). It was trained on movie plot summaries and metadata from the MovieLens-32M dataset, which were first converted into continuous embeddings using the `sentence-transformers/all-mpnet-base-v2` model.

The purpose of this RQ-VAE is to transform continuous, dense movie representations into a sequence of discrete tokens (SIDs), making them more interpretable, compressible, and suitable for various downstream tasks such as:
*   **Efficient Storage and Retrieval:** SIDs can be used as discrete identifiers for movies, enabling faster lookups and reducing storage requirements compared to full embeddings.
*   **Recommendation Systems:** SIDs can serve as features in recommendation models, potentially offering more explainable recommendations.
*   **Semantic Clustering:** The discrete nature of SIDs can help in grouping semantically similar movies.

## Model Architecture

The core of this model is the `ResidualQuantizer` from the `residual_quantized_vae.py` script. It consists of multiple layers of vector quantization, where each layer refines the quantization of the residual from the previous layer.

**Key Hyperparameters:**
*   **`num_layers`**: 4 (Number of residual quantization layers)
*   **`num_embeddings`**: 1024 (Number of distinct codebook vectors per layer)
*   **`embedding_dim`**: 768 (Dimensionality of the input embeddings and the codebook vectors)
*   **`commitment_cost`**: 0.25 (Weight for the commitment loss in the VQ-VAE objective)

The model was trained using PyTorch Lightning, optimizing for reconstruction loss (MSE between original and quantized embeddings) and a VQ-VAE loss component that encourages the embeddings to commit to codebook vectors.

## How to Use

This model can be loaded directly from the Hugging Face Hub using the `ResidualQuantizer.from_pretrained()` method.

```python
from residual_quantized_vae import ResidualQuantizer
import torch

# Load the pre-trained RQ-VAE model
# This will download the model weights from the Hugging Face Hub
rq_model = ResidualQuantizer.from_pretrained("krishnakamath/rq-vae-movielens")
rq_model.eval() # Set to evaluation mode

# Example: Quantize a dummy embedding
# In a real scenario, 'your_movie_embedding' would be a 768-dimensional tensor
# representing a movie (e.g., from a SentenceTransformer model).
your_movie_embedding = torch.randn(1, 768) # Batch size 1, embedding_dim 768

with torch.no_grad():
    quantized_embedding, semantic_ids, vq_loss = rq_model(your_movie_embedding)

print("Original Embedding Shape:", your_movie_embedding.shape)
print("Quantized Embedding Shape:", quantized_embedding.shape)
print("Semantic IDs (discrete tokens):", semantic_ids)
# semantic_ids will be a tensor of shape (batch_size, num_layers)
# Each element in semantic_ids corresponds to the index of the chosen codebook vector
# for that layer.
```

### Training Details

The model was trained using the `train.py` script with the following training hyperparameters:
*   **Learning Rate**: 1e-4
*   **Batch Size**: 128
*   **Epochs**: Up to 500 (with early stopping based on validation loss)

The training data consisted of embeddings generated from the `krishnakamath/movielens-32m-movies-enriched` dataset.

## Limitations and Future Work

*   **Interpretability:** While SIDs are discrete, their direct semantic meaning still requires further analysis and mapping.
*   **Generalization:** The model's performance is tied to the quality of the initial continuous embeddings and the diversity of the training data.
*   **Downstream Task Evaluation:** Further work is needed to rigorously evaluate the benefits of using SIDs in actual recommendation systems or semantic search applications.

## Citation

If you use this model or the associated code in your research, please consider citing the original MovieLens dataset:

F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19. https://doi.org/10.1145/2827872

## Acknowledgement
The Python scripts used to generate and process this model were developed with the assistance of Google's Gemini.
