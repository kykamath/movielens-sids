"""
pipeline.py

This module provides a set of classes to structure the ML pipeline for the SID project.
It follows an Object-Oriented approach to encapsulate the different stages of the workflow,
including configuration management, data handling, embedding generation, and model training.
"""

from dataclasses import dataclass, asdict
import os
import torch
from dotenv import load_dotenv
from huggingface_hub import login, HfApi
from datasets import load_dataset, Dataset
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from typing import Optional

from embedding_utils import generate_embeddings, get_embedding_column_name
from experiments import ExperimentConfig
from models import Movie
from rqvae_lightning import RQVAE, MovieEmbeddingDataModule
from residual_quantized_vae import ResidualQuantizer

# --- Configuration ---

@dataclass
class PipelineConfig:
    """Holds all configuration parameters for the pipeline, with support for experiment overrides."""
    # Hugging Face Hub IDs
    enriched_repo_id: str = "krishnakamath/movielens-32m-movies-enriched"
    sids_dataset_id: str = "krishnakamath/movielens-32m-movies-enriched-with-SIDs"
    hub_model_id: str = "krishnakamath/rq-vae-movielens"
    
    # Embedding Model
    embedding_model_name: str = 'sentence-transformers/all-mpnet-base-v2'
    
    # RQ-VAE Model Hyperparameters
    embedding_dim: int = 768
    num_layers: int = 4
    num_embeddings: int = 1024
    commitment_cost: float = 0.25
    
    # Training Hyperparameters
    learning_rate: float = 1e-4
    batch_size: int = 128
    epochs: int = 500
    
    # System and Experiment Config
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    experiment_name: Optional[str] = None
    dummy_run: bool = False
    smoke_test: bool = False
    max_data_samples: Optional[int] = None

    def __init__(self, experiment_name: Optional[str] = None, experiment_config: Optional[ExperimentConfig] = None, dummy_run: bool = False, smoke_test: bool = False):
        """
        Initializes the configuration, layering settings: defaults -> experiment -> smoke/dummy.
        """
        self.experiment_name = experiment_name
        self.dummy_run = dummy_run
        self.smoke_test = smoke_test

        # 1. Apply experiment overrides first
        if experiment_config:
            print(f"Applying config for experiment: {experiment_name}")
            for key, value in asdict(experiment_config).items():
                if hasattr(self, key):
                    setattr(self, key, value)
        
        # 2. Apply smoke test overrides
        if self.smoke_test:
            print("--- SMOKE TEST ACTIVATED ---")
            self.epochs = 2
            self.batch_size = 8
            self.max_data_samples = 64
            if self.hub_model_id:
                self.hub_model_id += "-smoke-test"
            if self.experiment_name:
                self.experiment_name += "-smoke-test"
            else:
                self.experiment_name = "smoke-test"

        # 3. Dummy run overrides everything
        if self.dummy_run:
            print("--- DUMMY RUN ACTIVATED ---")
            self.epochs = 1
            self.batch_size = 4
            self.max_data_samples = 16
            if self.experiment_name and not self.smoke_test:
                 self.experiment_name = f"{self.experiment_name}-dummy"
            else:
                 self.experiment_name = "dummy"


        self._handle_auth()

    def _handle_auth(self):
        """Handle authentication after initialization."""
        load_dotenv()
        self.hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
        # Log in if we have a token and this is NOT a dummy run
        if self.hf_token and not self.dummy_run:
            print("Logging in to Hugging Face Hub...")
            login(token=self.hf_token)
        elif self.dummy_run:
            print("Dummy run: Skipping Hugging Face login.")
        else:
            print("Warning: HUGGING_FACE_HUB_TOKEN not found. Uploads will be skipped.")

# --- Pipeline Components ---

class DatasetManager:
    """Handles loading and preparation of datasets."""
    def __init__(self, config: PipelineConfig):
        self.config = config

    def load_source_dataset(self) -> Dataset:
        """Loads the source enriched dataset from the Hub, slicing if needed."""
        print(f"Loading source dataset from: {self.config.enriched_repo_id}")
        dataset = load_dataset(self.config.enriched_repo_id, split="train")
        if self.config.max_data_samples:
            print(f"Slicing dataset to {self.config.max_data_samples} samples.")
            dataset = dataset.select(range(self.config.max_data_samples))
        return dataset

    def create_final_dataset(self, source_dataset: Dataset, embeddings: list, sids: list, items_with_plots_indices: list) -> Dataset:
        """Combines source data with new embeddings and SIDs to create the final dataset."""
        embedding_column_name = get_embedding_column_name(self.config.embedding_model_name)
        print(f"Using new embedding column name: '{embedding_column_name}'")

        movie_id_to_embedding = {source_dataset[i]['movie_id']: emb for i, emb in zip(items_with_plots_indices, embeddings)}
        movie_id_to_sid = {source_dataset[i]['movie_id']: sid for i, sid in zip(items_with_plots_indices, sids)}

        def add_embeddings_and_sids(example):
            movie_id = example['movie_id']
            example[embedding_column_name] = movie_id_to_embedding.get(movie_id, [])
            example['semantic_id'] = movie_id_to_sid.get(movie_id, [])
            return example

        final_dataset = source_dataset.map(add_embeddings_and_sids)
        
        columns_to_remove = [col for col in final_dataset.column_names if 'embedding' in col and col != embedding_column_name]
        if columns_to_remove:
            print(f"Removing old embedding columns: {columns_to_remove}")
            final_dataset = final_dataset.remove_columns(columns_to_remove)
            
        return final_dataset

    def push_to_hub(self, dataset: Dataset):
        """Pushes a dataset to the Hub, skipping if in dummy mode."""
        if self.config.dummy_run:
            print("Dummy run: Skipping dataset upload to Hugging Face Hub.")
            return
            
        print(f"\nUploading final dataset to: {self.config.sids_dataset_id}")
        dataset.push_to_hub(self.config.sids_dataset_id, private=False)
        print(f"✅ Dataset available at: https://huggingface.co/datasets/{self.config.sids_dataset_id}")


class EmbeddingManager:
    """Manages the generation of embeddings."""
    def __init__(self, config: PipelineConfig):
        self.config = config

    def generate(self, source_dataset: Dataset) -> (torch.Tensor, list, list):
        """Generates embeddings for the movie data."""
        texts_to_embed = []
        items_with_plots_indices = []
        
        print("Preparing texts for embedding generation...")
        for i, item in enumerate(source_dataset):
            if item.get('plot_summary'):
                movie = Movie(movie_id=item.get('movie_id'), title=item.get('title'), genres=item.get('genres', []), plot_summary=item.get('plot_summary', ''), director=item.get('director', ''), stars=item.get('stars', []))
                texts_to_embed.append(movie.to_embedding_string())
                items_with_plots_indices.append(i)

        if not texts_to_embed:
            raise ValueError("No valid texts found to generate embeddings.")

        embeddings = generate_embeddings(model_name=self.config.embedding_model_name, texts_to_embed=texts_to_embed, device=self.config.device)
        return torch.tensor(embeddings, dtype=torch.float32).to(self.config.device), items_with_plots_indices, embeddings.tolist()


class RQVAEOrchestrator:
    """Manages the training, loading, and inference of the RQ-VAE model."""
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.model = None

    def train(self):
        """Initializes and trains the RQ-VAE model."""
        print("--- Generating Embeddings for Training ---")
        dataset_manager = DatasetManager(self.config)
        embedding_manager = EmbeddingManager(self.config)
        source_dataset = dataset_manager.load_source_dataset()
        embeddings_tensor, _, _ = embedding_manager.generate(source_dataset)

        data_module = MovieEmbeddingDataModule(
            embeddings_tensor=embeddings_tensor,
            batch_size=self.config.batch_size
        )

        self.model = RQVAE(num_layers=self.config.num_layers, num_embeddings=self.config.num_embeddings, embedding_dim=self.config.embedding_dim, commitment_cost=self.config.commitment_cost, learning_rate=self.config.learning_rate)

        log_name = self.config.experiment_name or "default"
        logger = TensorBoardLogger("tb_logs", name="rq_vae_model", version=log_name)
        
        checkpoint_dir = f"checkpoints/{log_name}/"
        checkpoint = ModelCheckpoint(monitor='val_loss', dirpath=checkpoint_dir, filename='best-model', save_top_k=1, mode='min')
        
        early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=True, mode='min')

        trainer = Trainer(max_epochs=self.config.epochs, accelerator="auto", callbacks=[early_stop, checkpoint], logger=logger)

        print(f"--- Starting Training for Experiment: {log_name} ---")
        trainer.fit(self.model, datamodule=data_module)
        print(f"\n--- Training Complete for Experiment: {log_name} ---")
        
        if self.config.hf_token and checkpoint.best_model_path and not self.config.dummy_run:
            self._publish(checkpoint)
        else:
            print("\nDummy run or token not found: Skipping model upload.")

    def _publish(self, checkpoint_callback: ModelCheckpoint):
        """Uploads the best model to the Hub and tags it with the experiment name."""
        print(f"\nUploading best model from '{checkpoint_callback.best_model_path}' to Hub: {self.config.hub_model_id}")
        best_model = RQVAE.load_from_checkpoint(checkpoint_callback.best_model_path)
        quantizer_model = best_model.quantizer
        
        commit_info = quantizer_model.push_to_hub(
            repo_id=self.config.hub_model_id, 
            commit_message=f"Upload best model for experiment {self.config.experiment_name}"
        )
        print(f"✅ Model successfully uploaded to {self.config.hub_model_id}")

        if self.config.experiment_name:
            print(f"Tagging commit with experiment name: '{self.config.experiment_name}'")
            api = HfApi()
            api.create_tag(
                repo_id=self.config.hub_model_id,
                tag=self.config.experiment_name,
                tag_message=f"Model for experiment '{self.config.experiment_name}'",
                revision=commit_info.oid,
                repo_type="model"
            )
            print(f"✅ Successfully tagged commit with '{self.config.experiment_name}'")

    def load_from_hub(self, revision: Optional[str] = None) -> ResidualQuantizer:
        """Loads a pre-trained model. In dummy mode, initializes a new model instead."""
        if self.config.dummy_run:
            print("Dummy run: Initializing a new, untrained model for inference.")
            dummy_model = RQVAE(
                num_layers=self.config.num_layers, 
                num_embeddings=self.config.num_embeddings, 
                embedding_dim=self.config.embedding_dim,
                commitment_cost=self.config.commitment_cost,
                learning_rate=self.config.learning_rate
            ).quantizer
            return dummy_model.to(self.config.device)

        revision = revision or self.config.experiment_name or "main"
        print(f"Loading pre-trained RQ-VAE model from: {self.config.hub_model_id} at revision '{revision}'")
        try:
            model = ResidualQuantizer.from_pretrained(self.config.hub_model_id, revision=revision)
            model = model.to(self.config.device)
            model.eval()
            return model
        except Exception as e:
            raise RuntimeError(f"Could not load model from Hub. Error: {e}")

    def generate_sids(self, model: ResidualQuantizer, embeddings_tensor: torch.Tensor) -> list:
        """Generates Semantic IDs from embeddings using the pre-trained model."""
        print("Generating Semantic IDs from embeddings...")
        with torch.no_grad():
            _, sids_tensor, _ = model(embeddings_tensor)
            return sids_tensor.cpu().numpy().tolist()
