import argparse
import os
import sys
import time
import json
import torch
import numpy as np

from config import DataConfig, ModelConfig, TrainConfig
from data import prepare_data, get_dataloader
from train import train_model, train_all_models
from evaluate import evaluate_model, generate_evaluation_report
from models import count_parameters, build_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Protein Structure Prediction from Sequence"
    )

    parser.add_argument('--synthetic', action='store_true',
                        help='Use synthetic data')
    parser.add_argument('--num-proteins', type=int, default=800)
    parser.add_argument('--max-seq-len', type=int, default=200)
    parser.add_argument('--skip-download', action='store_true')

    parser.add_argument('--arch', type=str, default='all',
                        choices=['all', 'transformer', 'cnn', 'bilstm'])
    parser.add_argument('--embed-dim', type=int, default=128)

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--output-dir', type=str, default='results')

    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    args = parse_args()
    set_seed(args.seed)

    print("=" * 60)
    print("  Protein Structure Prediction from Sequence")
    print("  Comparing: Transformer / 1D CNN / BiLSTM")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    data_config = DataConfig(
        num_proteins=args.num_proteins,
        max_seq_length=args.max_seq_len,
    )
    model_config = ModelConfig(
        embed_dim=args.embed_dim,
        max_seq_len=args.max_seq_len,
    )
    train_config = TrainConfig(
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        patience=args.patience,
        device=device,
        num_workers=0 if args.synthetic else 4,
    )

    print("\n--- Data Preparation ---")
    t0 = time.time()
    train_data, val_data, test_data = prepare_data(
        data_config,
        use_synthetic=args.synthetic,
        seed=args.seed,
    )
    print(f"Data ready in {time.time()-t0:.1f}s")

    max_len = args.max_seq_len
    if args.synthetic:
        max_len = min(max_len, 100)

    train_loader = get_dataloader(
        train_data, max_len=max_len,
        batch_size=args.batch_size, num_workers=train_config.num_workers,
    )
    val_loader = get_dataloader(
        val_data, max_len=max_len,
        batch_size=args.batch_size, shuffle=False,
        num_workers=train_config.num_workers,
    )
    test_loader = get_dataloader(
        test_data, max_len=max_len,
        batch_size=args.batch_size, shuffle=False,
        num_workers=train_config.num_workers,
    )

    print("\n--- Model Summary ---")
    for arch in ['transformer', 'cnn', 'bilstm']:
        m = build_model(arch, model_config)
        print(f"  {m.name:<12s}: {count_parameters(m):>10,} parameters")
    del m

    print("\n--- Training ---")
    architectures = (
        ['transformer', 'cnn', 'bilstm'] if args.arch == 'all'
        else [args.arch]
    )

    trained_models = {}
    histories = {}

    for arch in architectures:
        #bilstm needs higher lr to avoid collapse
        if arch == 'bilstm':
            import copy
            bilstm_config = copy.deepcopy(train_config)
            bilstm_config.learning_rate = args.lr * 10
            arch_config = bilstm_config
        else:
            arch_config = train_config

        model, history = train_model(
            arch, train_loader, val_loader,
            model_config, arch_config,
        )
        trained_models[arch] = model
        histories[arch] = history

    print("\n--- Evaluation on Test Set ---")
    all_results = {}
    for arch, model in trained_models.items():
        print(f"\nEvaluating {arch}...")
        results = evaluate_model(model, test_loader, device=device)
        all_results[arch] = results
        print(f"  RMSD: {results['rmsd_mean']:.2f} ± {results['rmsd_std']:.2f} Å "
              f"(median: {results['rmsd_median']:.2f})")
        print(f"  DME:  {results['dme_mean']:.2f} ± {results['dme_std']:.2f} Å "
              f"(median: {results['dme_median']:.2f})")

    print("\n--- Generating Report ---")
    generate_evaluation_report(all_results, histories, output_dir=args.output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()