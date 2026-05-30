import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist, squareform

from config import TrainConfig, ModelConfig


def classical_mds(D, n_components=3):
    #classical metric mds - find coords from distance matrix
    N = D.shape[0]
    D_sq = D ** 2

    H = np.eye(N) - np.ones((N, N)) / N
    B = -0.5 * H @ D_sq @ H

    eigenvalues, eigenvectors = np.linalg.eigh(B)

    idx = np.argsort(eigenvalues)[::-1][:n_components]
    L = np.sqrt(np.maximum(eigenvalues[idx], 0))
    V = eigenvectors[:, idx]

    X = V * L[np.newaxis, :]
    return X.astype(np.float64)


def kabsch_align(P, Q):
    #align P onto Q, returns aligned P and centered Q
    P_mean = P.mean(axis=0)
    Q_mean = Q.mean(axis=0)
    P_c = P - P_mean
    Q_c = Q - Q_mean

    H = P_c.T @ Q_c
    U, S, Vt = np.linalg.svd(H)

    #correct for reflection
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1.0, 1.0, np.sign(d)])

    R = Vt.T @ sign_matrix @ U.T
    P_aligned = P_c @ R.T

    return P_aligned, Q_c


def compute_rmsd(pred_coords, true_coords, align=True):
    if align:
        pred_coords, true_coords = kabsch_align(pred_coords, true_coords)

    diff = pred_coords - true_coords
    rmsd = np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))
    return float(rmsd)


def compute_dme(pred_dist, true_dist, mask=None):
    #mean absolute difference between predicted and true distances
    diff = np.abs(pred_dist - true_dist)
    if mask is not None:
        upper = np.triu(mask, k=1)
        if upper.sum() == 0:
            return 0.0
        return float(diff[upper > 0].mean())
    else:
        N = pred_dist.shape[0]
        upper_idx = np.triu_indices(N, k=1)
        return float(diff[upper_idx].mean())


@torch.no_grad()
def evaluate_model(model, dataloader, device='cpu', max_samples=None):
    #predict distance matrices, reconstruct 3d via mds, compute rmsd and dme
    model.eval()
    model.to(device)

    results = []
    count = 0

    for batch in dataloader:
        tokens = batch['tokens'].to(device)
        target_dist = batch['distance_matrix']
        mask = batch['mask']
        lengths = batch['length']
        true_coords = batch['coords']

        pred_dist = model(tokens, mask.to(device)).cpu().numpy()

        B = tokens.size(0)
        for i in range(B):
            if max_samples and count >= max_samples:
                break

            L = lengths[i].item()
            pred_d = pred_dist[i, :L, :L]
            true_d = target_dist[i, :L, :L].numpy()
            true_c = true_coords[i, :L].numpy()

            try:
                pred_c = classical_mds(pred_d, n_components=3)
            except Exception:
                pred_c = np.zeros((L, 3))

            rmsd = compute_rmsd(pred_c, true_c, align=True)
            dme = compute_dme(pred_d, true_d)

            results.append({
                'length': L,
                'rmsd': rmsd,
                'dme': dme,
                'pred_dist': pred_d,
                'true_dist': true_d,
                'pred_coords': pred_c,
                'true_coords': true_c,
            })
            count += 1

        if max_samples and count >= max_samples:
            break

    rmsds = [r['rmsd'] for r in results]
    dmes = [r['dme'] for r in results]

    summary = {
        'n_proteins': len(results),
        'rmsd_mean': float(np.mean(rmsds)),
        'rmsd_std': float(np.std(rmsds)),
        'rmsd_median': float(np.median(rmsds)),
        'dme_mean': float(np.mean(dmes)),
        'dme_std': float(np.std(dmes)),
        'dme_median': float(np.median(dmes)),
        'per_protein': results,
    }

    return summary


def plot_training_curves(histories, save_path="training_curves.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {'transformer': '#2196F3', 'cnn': '#4CAF50', 'bilstm': '#FF9800'}

    for name, history in histories.items():
        color = colors.get(name.lower(), '#999999')
        epochs = range(1, len(history['train_loss']) + 1)

        axes[0].plot(epochs, history['train_loss'], color=color,
                     label=name, linewidth=2)
        axes[1].plot(epochs, history['val_loss'], color=color,
                     label=name, linewidth=2)

    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MSE Loss')
    axes[1].set_title('Validation Loss')
    axes[1].legend()
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved training curves to {save_path}")


def plot_distance_matrices(pred_dist, true_dist, title="", save_path="distance_matrix.png"):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    vmax = max(true_dist.max(), pred_dist.max())

    im0 = axes[0].imshow(true_dist, cmap='viridis', vmin=0, vmax=vmax)
    axes[0].set_title('Ground Truth')
    axes[0].set_xlabel('Residue index')
    axes[0].set_ylabel('Residue index')
    plt.colorbar(im0, ax=axes[0], label='Distance (Å)')

    im1 = axes[1].imshow(pred_dist, cmap='viridis', vmin=0, vmax=vmax)
    axes[1].set_title('Predicted')
    axes[1].set_xlabel('Residue index')
    plt.colorbar(im1, ax=axes[1], label='Distance (Å)')

    error = np.abs(pred_dist - true_dist)
    im2 = axes[2].imshow(error, cmap='Reds')
    axes[2].set_title('Absolute Error')
    axes[2].set_xlabel('Residue index')
    plt.colorbar(im2, ax=axes[2], label='Error (Å)')

    if title:
        fig.suptitle(title, fontsize=14, y=1.02)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_3d_structure(pred_coords, true_coords, title="", save_path="structure_3d.png"):
    pred_aligned, true_centered = kabsch_align(pred_coords, true_coords)

    fig = plt.figure(figsize=(14, 6))

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(*true_centered.T, 'b-', linewidth=1.5, alpha=0.7)
    ax1.scatter(*true_centered.T, c=np.arange(len(true_centered)),
                cmap='viridis', s=20, zorder=5)
    ax1.set_title('Ground Truth')
    ax1.set_xlabel('X (Å)')
    ax1.set_ylabel('Y (Å)')
    ax1.set_zlabel('Z (Å)')

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot(*pred_aligned.T, 'r-', linewidth=1.5, alpha=0.7, label='Predicted')
    ax2.plot(*true_centered.T, 'b-', linewidth=1.0, alpha=0.3, label='True')
    ax2.scatter(*pred_aligned.T, c=np.arange(len(pred_aligned)),
                cmap='plasma', s=20, zorder=5)
    ax2.set_title('Predicted (aligned)')
    ax2.set_xlabel('X (Å)')
    ax2.set_ylabel('Y (Å)')
    ax2.set_zlabel('Z (Å)')
    ax2.legend()

    if title:
        fig.suptitle(title, fontsize=14, y=1.02)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_rmsd_distribution(all_results, save_path="rmsd_distribution.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {'transformer': '#2196F3', 'cnn': '#4CAF50', 'bilstm': '#FF9800'}

    names = []
    rmsd_data = []
    dme_data = []

    for name, summary in all_results.items():
        rmsds = [r['rmsd'] for r in summary['per_protein']]
        dmes = [r['dme'] for r in summary['per_protein']]
        names.append(name)
        rmsd_data.append(rmsds)
        dme_data.append(dmes)

    bp = axes[0].boxplot(rmsd_data, labels=names, patch_artist=True)
    for i, (patch, name) in enumerate(zip(bp['boxes'], names)):
        patch.set_facecolor(colors.get(name.lower(), '#999999'))
        patch.set_alpha(0.7)
    axes[0].set_ylabel('RMSD (Å)')
    axes[0].set_title('RMSD Distribution by Model')
    axes[0].grid(True, alpha=0.3, axis='y')

    bp2 = axes[1].boxplot(dme_data, labels=names, patch_artist=True)
    for i, (patch, name) in enumerate(zip(bp2['boxes'], names)):
        patch.set_facecolor(colors.get(name.lower(), '#999999'))
        patch.set_alpha(0.7)
    axes[1].set_ylabel('DME (Å)')
    axes[1].set_title('DME Distribution by Model')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved RMSD/DME distribution to {save_path}")


def plot_rmsd_vs_length(all_results, save_path="rmsd_vs_length.png"):
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {'transformer': '#2196F3', 'cnn': '#4CAF50', 'bilstm': '#FF9800'}

    for name, summary in all_results.items():
        lengths = [r['length'] for r in summary['per_protein']]
        rmsds = [r['rmsd'] for r in summary['per_protein']]
        color = colors.get(name.lower(), '#999999')
        ax.scatter(lengths, rmsds, c=color, alpha=0.5, s=30, label=name)

    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('RMSD (Å)')
    ax.set_title('RMSD vs. Sequence Length')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved RMSD vs. length to {save_path}")


def generate_evaluation_report(all_results, histories, output_dir="results"):
    #generate all plots and summary
    os.makedirs(output_dir, exist_ok=True)

    plot_training_curves(histories, os.path.join(output_dir, "training_curves.png"))
    plot_rmsd_distribution(all_results, os.path.join(output_dir, "rmsd_dme_distribution.png"))
    plot_rmsd_vs_length(all_results, os.path.join(output_dir, "rmsd_vs_length.png"))

    #best and worst case for each model
    for name, summary in all_results.items():
        proteins = summary['per_protein']
        if not proteins:
            continue

        rmsds = [p['rmsd'] for p in proteins]
        best_idx = int(np.argmin(rmsds))
        worst_idx = int(np.argmax(rmsds))

        for label, idx in [('best', best_idx), ('worst', worst_idx)]:
            p = proteins[idx]
            prefix = f"{name}_{label}"

            plot_distance_matrices(
                p['pred_dist'], p['true_dist'],
                title=f"{name} — {label} case (RMSD={p['rmsd']:.2f} Å, L={p['length']})",
                save_path=os.path.join(output_dir, f"{prefix}_distmat.png"),
            )
            plot_3d_structure(
                p['pred_coords'], p['true_coords'],
                title=f"{name} — {label} case (RMSD={p['rmsd']:.2f} Å, L={p['length']})",
                save_path=os.path.join(output_dir, f"{prefix}_structure.png"),
            )

    #print summary table
    print("\n" + "=" * 70)
    print(f"{'Model':<15} {'RMSD Mean':>10} {'RMSD Med':>10} {'RMSD Std':>10} "
          f"{'DME Mean':>10} {'DME Med':>10}")
    print("-" * 70)
    summary_data = {}
    for name, summary in all_results.items():
        print(f"{name:<15} {summary['rmsd_mean']:>10.2f} {summary['rmsd_median']:>10.2f} "
              f"{summary['rmsd_std']:>10.2f} {summary['dme_mean']:>10.2f} "
              f"{summary['dme_median']:>10.2f}")
        summary_data[name] = {
            'rmsd_mean': summary['rmsd_mean'],
            'rmsd_median': summary['rmsd_median'],
            'rmsd_std': summary['rmsd_std'],
            'dme_mean': summary['dme_mean'],
            'dme_median': summary['dme_median'],
            'n_proteins': summary['n_proteins'],
        }
    print("=" * 70)

    import json
    with open(os.path.join(output_dir, "summary.json"), 'w') as f:
        json.dump(summary_data, f, indent=2)
    print(f"\nAll results saved to {output_dir}/")


if __name__ == "__main__":
    np.random.seed(42)

    N = 50
    t = np.linspace(0, 4 * np.pi, N)
    true_coords = np.column_stack([
        5 * np.cos(t), 5 * np.sin(t), t * 2
    ])

    true_dist = squareform(pdist(true_coords))

    noise = np.random.randn(N, N) * 1.0
    noisy_dist = true_dist + (noise + noise.T) / 2
    np.fill_diagonal(noisy_dist, 0)
    noisy_dist = np.maximum(noisy_dist, 0)

    recon_coords = classical_mds(noisy_dist, 3)

    rmsd = compute_rmsd(recon_coords, true_coords, align=True)
    dme = compute_dme(noisy_dist, true_dist)

    print(f"MDS+Kabsch test: RMSD = {rmsd:.2f} Å, DME = {dme:.2f} Å")