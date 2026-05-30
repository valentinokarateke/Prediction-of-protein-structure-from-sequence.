import os
import torch
import numpy as np

from config import DataConfig, ModelConfig, TrainConfig
from data import prepare_data, get_dataloader
from models import build_model
from evaluate import evaluate_model, generate_evaluation_report


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    data_config = DataConfig(num_proteins=800, max_seq_length=200)
    model_config = ModelConfig(embed_dim=128, max_seq_len=200)

    train_data, val_data, test_data = prepare_data(data_config, use_synthetic=False, seed=42)
    test_loader = get_dataloader(
        test_data, max_len=200, batch_size=8, shuffle=False, num_workers=0
    )

    trained_models = {}
    histories = {}

    for arch in ['transformer', 'cnn', 'bilstm']:
        checkpoint_path = os.path.join("checkpoints", f"best_{arch}.pt")
        if not os.path.exists(checkpoint_path):
            print(f"  Skipping {arch} — no checkpoint found at {checkpoint_path}")
            continue

        print(f"Loading {arch} from {checkpoint_path}...")
        model = build_model(arch, model_config)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        trained_models[arch] = model
        histories[arch] = checkpoint['history']

    print("\n--- Evaluation on Test Set ---")
    all_results = {}
    for arch, model in trained_models.items():
        print(f"Evaluating {arch}...")
        results = evaluate_model(model, test_loader, device=device)
        all_results[arch] = results
        print(f"  RMSD: {results['rmsd_mean']:.2f} ± {results['rmsd_std']:.2f} Å "
              f"(median: {results['rmsd_median']:.2f})")
        print(f"  DME:  {results['dme_mean']:.2f} ± {results['dme_std']:.2f} Å "
              f"(median: {results['dme_median']:.2f})")

    print("\n--- Generating Combined Report ---")
    generate_evaluation_report(all_results, histories, output_dir="results")
    print("Done!")


if __name__ == "__main__":
    main()