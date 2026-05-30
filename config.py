from dataclasses import dataclass, field
from typing import Optional
import torch


@dataclass
class DataConfig:
    data_dir: str = "pdb_data"
    processed_dir: str = "processed_data"
    max_resolution: float = 2.5
    min_seq_length: int = 30
    max_seq_length: int = 200
    num_proteins: int = 800
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    sequence_identity_threshold: float = 0.3


@dataclass
class ModelConfig:
    vocab_size: int = 21                 #20 amino acids + padding
    embed_dim: int = 128
    max_seq_len: int = 200
    dropout: float = 0.1

    #transformer
    transformer_layers: int = 4
    transformer_heads: int = 8
    transformer_ff_dim: int = 512

    #cnn
    cnn_layers: int = 6
    cnn_kernel_size: int = 5
    cnn_channels: int = 128

    #bilstm
    lstm_layers: int = 2
    lstm_hidden_dim: int = 128

    #output head
    head_hidden_dim: int = 64


@dataclass
class TrainConfig:
    batch_size: int = 8
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    patience: int = 15
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 5
    gradient_clip: float = 1.0
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    seed: int = 42


#amino acid vocab, 0=padding 1-20=amino acids
AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: i + 1 for i, aa in enumerate(AA_LIST)}
IDX_TO_AA = {i + 1: aa for i, aa in enumerate(AA_LIST)}

#three letter to one letter mapping
THREE_TO_ONE = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
}