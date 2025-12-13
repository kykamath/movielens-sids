import torch
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset, random_split
from residual_quantized_vae import ResidualQuantizer

class RQVAE(pl.LightningModule):
    """
    The PyTorch Lightning module that wraps the ResidualQuantizer for training.
    """
    def __init__(self, num_layers, num_embeddings, embedding_dim, commitment_cost, learning_rate):
        super().__init__()
        self.save_hyperparameters()
        self.quantizer = ResidualQuantizer(
            num_layers=self.hparams.num_layers,
            num_embeddings=self.hparams.num_embeddings,
            embedding_dim=self.hparams.embedding_dim,
            commitment_cost=self.hparams.commitment_cost
        )

    def forward(self, z):
        return self.quantizer(z)

    def _common_step(self, batch, batch_idx):
        batch_z, = batch
        z_quantized, _, vq_loss = self.quantizer(batch_z)
        reconstruction_loss = F.mse_loss(z_quantized, batch_z)
        total_loss = reconstruction_loss + vq_loss
        return total_loss, reconstruction_loss, vq_loss

    def training_step(self, batch, batch_idx):
        total_loss, recon_loss, vq_loss = self._common_step(batch, batch_idx)
        self.log('train_loss', total_loss, prog_bar=True)
        self.log('train_recon_loss', recon_loss, on_epoch=True, on_step=False)
        self.log('train_vq_loss', vq_loss, on_epoch=True, on_step=False)
        return total_loss

    def validation_step(self, batch, batch_idx):
        total_loss, recon_loss, vq_loss = self._common_step(batch, batch_idx)
        self.log('val_loss', total_loss, prog_bar=True)
        self.log('val_recon_loss', recon_loss, on_epoch=True)
        self.log('val_vq_loss', vq_loss, on_epoch=True)
        return total_loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


class MovieEmbeddingDataModule(pl.LightningDataModule):
    """
    A PyTorch Lightning DataModule that works with a pre-generated tensor of embeddings.
    """
    def __init__(self, embeddings_tensor: torch.Tensor, batch_size: int, num_workers: int = 0):
        """
        Initializes the DataModule.
        
        Args:
            embeddings_tensor: The tensor containing all embeddings.
            batch_size: The batch size for the dataloaders.
            num_workers: The number of worker processes for data loading. 
                         Defaults to 0 to prevent CUDA forking issues in certain environments.
        """
        super().__init__()
        self.embeddings_tensor = embeddings_tensor
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: str = None):
        """Splits the provided embedding tensor into training and validation sets."""
        full_dataset = TensorDataset(self.embeddings_tensor)
        
        print(f"Total samples in dataset: {len(full_dataset)}")
        
        # Split the data
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])
        print(f"Data split into {len(self.train_dataset)} training and {len(self.val_dataset)} validation samples.")

    def train_dataloader(self):
        # persistent_workers is only valid for num_workers > 0
        persistent = self.num_workers > 0
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=persistent)

    def val_dataloader(self):
        persistent = self.num_workers > 0
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=persistent)
