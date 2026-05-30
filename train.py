import os
import time
import json
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from config import TrainConfig, ModelConfig
from models import build_model, count_parameters


def masked_mse_loss(pred, target, mask):
    #mse loss only over valid non-padded pairs
    diff_sq = (pred - target) ** 2
    masked_diff = diff_sq * mask
    loss = masked_diff.sum() / mask.sum().clamp(min=1)
    return loss


class Trainer:
    def __init__(self, model, train_config):
        self.model = model.to(train_config.device)
        self.config = train_config
        self.device = train_config.device

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=train_config.learning_rate,
            weight_decay=train_config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=train_config.lr_scheduler_factor,
            patience=train_config.lr_scheduler_patience,
        )

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'epoch_time': [],
        }
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            tokens = batch['tokens'].to(self.device)
            target = batch['distance_matrix'].to(self.device)
            mask = batch['mask'].to(self.device)

            self.optimizer.zero_grad()
            pred = self.model(tokens, mask)
            loss = masked_mse_loss(pred, target, mask)
            loss.backward()

            if self.config.gradient_clip > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip
                )

            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            tokens = batch['tokens'].to(self.device)
            target = batch['distance_matrix'].to(self.device)
            mask = batch['mask'].to(self.device)

            pred = self.model(tokens, mask)
            loss = masked_mse_loss(pred, target, mask)

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def save_checkpoint(self, path):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history,
        }, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']

    def fit(self, train_loader, val_loader):
        model_name = getattr(self.model, 'name', 'model')
        n_params = count_parameters(self.model)
        print(f"\n{'='*60}")
        print(f"Training {model_name} ({n_params:,} parameters)")
        print(f"Device: {self.device}")
        print(f"{'='*60}")

        checkpoint_path = os.path.join(
            self.config.checkpoint_dir, f"best_{model_name.lower()}.pt"
        )

        for epoch in range(1, self.config.num_epochs + 1):
            t0 = time.time()

            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            epoch_time = time.time() - t0
            lr = self.optimizer.param_groups[0]['lr']

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(lr)
            self.history['epoch_time'].append(epoch_time)

            self.scheduler.step(val_loss)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(checkpoint_path)
                marker = " ★ (saved)"
            else:
                self.patience_counter += 1
                marker = ""

            if epoch % 1 == 0 or epoch == 1:
                print(
                    f"  Epoch {epoch:3d}/{self.config.num_epochs} | "
                    f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                    f"LR: {lr:.2e} | Time: {epoch_time:.1f}s{marker}"
                )

            if self.patience_counter >= self.config.patience:
                print(f"\n  Early stopping at epoch {epoch} "
                      f"(no improvement for {self.config.patience} epochs)")
                break

        if os.path.exists(checkpoint_path):
            self.load_checkpoint(checkpoint_path)
            print(f"  Loaded best model (val loss: {self.best_val_loss:.4f})")

        return self.history


def train_model(architecture, train_loader, val_loader,
                model_config=None, train_config=None):
    if model_config is None:
        model_config = ModelConfig()
    if train_config is None:
        train_config = TrainConfig()

    model = build_model(architecture, model_config)
    trainer = Trainer(model, train_config)
    history = trainer.fit(train_loader, val_loader)

    return trainer.model, history


def train_all_models(train_loader, val_loader,
                     model_config=None, train_config=None):
    results = {}
    for arch in ['transformer', 'cnn', 'bilstm']:
        model, history = train_model(
            arch, train_loader, val_loader, model_config, train_config
        )
        results[arch] = {'model': model, 'history': history}
    return results


if __name__ == "__main__":
    from data import prepare_data, get_dataloader
    from config import DataConfig

    data_config = DataConfig(num_proteins=50, max_seq_length=60)
    train_config = TrainConfig(
        batch_size=4, num_epochs=5, learning_rate=1e-3,
        num_workers=0, patience=3,
    )
    model_config = ModelConfig(max_seq_len=60)

    train_data, val_data, _ = prepare_data(data_config, use_synthetic=True)
    train_loader = get_dataloader(train_data, max_len=60, batch_size=4, num_workers=0)
    val_loader = get_dataloader(val_data, max_len=60, batch_size=4, shuffle=False, num_workers=0)

    model, history = train_model('transformer', train_loader, val_loader,
                                 model_config, train_config)
    print(f"\nFinal val loss: {history['val_loss'][-1]:.4f}")