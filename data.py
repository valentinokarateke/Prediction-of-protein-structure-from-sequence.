import os
import json
import pickle
import requests
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

from config import (
    DataConfig, AA_TO_IDX, THREE_TO_ONE
)


def search_pdb(config: DataConfig) -> list:
    #query rcsb pdb for single-chain xray structures
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "exptl.method",
                        "operator": "exact_match",
                        "value": "X-RAY DIFFRACTION"
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.resolution_combined",
                        "operator": "less_or_equal",
                        "value": config.max_resolution
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_assembly_info.polymer_entity_count",
                        "operator": "equals",
                        "value": 1
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.deposited_polymer_monomer_count",
                        "operator": "range",
                        "value": {
                            "from": config.min_seq_length,
                            "to": config.max_seq_length,
                            "include_lower": True,
                            "include_upper": True,
                        }
                    }
                }
            ]
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": config.num_proteins * 2},
            "sort": [{"sort_by": "score", "direction": "desc"}]
        }
    }

    url = "https://search.rcsb.org/rcsbsearch/v2/query"
    print(f"Querying RCSB PDB for up to {config.num_proteins} structures...")
    response = requests.post(url, json=query, timeout=60)
    response.raise_for_status()
    results = response.json()
    pdb_ids = [hit["identifier"] for hit in results.get("result_set", [])]
    print(f"  Found {len(pdb_ids)} candidate PDB entries.")
    return pdb_ids[:config.num_proteins * 2]


def download_and_parse(pdb_ids: list, config: DataConfig) -> list:
    #download pdb files and extract ca coords and sequences
    from Bio.PDB import PDBList, PDBParser

    os.makedirs(config.data_dir, exist_ok=True)
    pdb_list = PDBList(verbose=False)
    parser = PDBParser(QUIET=True)

    dataset = []
    target = config.num_proteins

    for pdb_id in tqdm(pdb_ids, desc="Downloading & parsing"):
        if len(dataset) >= target:
            break
        try:
            filename = pdb_list.retrieve_pdb_file(
                pdb_id, file_format='pdb', pdir=config.data_dir, overwrite=False
            )
            if filename is None or not os.path.exists(filename):
                continue

            structure = parser.get_structure(pdb_id, filename)
            model = structure[0]
            chain = list(model.get_chains())[0]

            ca_coords = []
            sequence = []

            for residue in chain.get_residues():
                #skip hetero atoms and water
                hetflag = residue.get_id()[0]
                if hetflag != ' ':
                    continue
                resname = residue.get_resname().strip()
                if resname not in THREE_TO_ONE:
                    continue
                if 'CA' not in residue:
                    continue

                ca_coords.append(residue['CA'].get_vector().get_array())
                sequence.append(THREE_TO_ONE[resname])

            seq_len = len(sequence)
            if config.min_seq_length <= seq_len <= config.max_seq_length:
                dataset.append({
                    'pdb_id': pdb_id,
                    'sequence': ''.join(sequence),
                    'coords': np.array(ca_coords, dtype=np.float32),
                })
        except Exception as e:
            continue

    print(f"Successfully parsed {len(dataset)} proteins.")
    return dataset


def compute_distance_matrix(coords: np.ndarray) -> np.ndarray:
    return squareform(pdist(coords, metric='euclidean')).astype(np.float32)


def encode_sequence(sequence: str) -> np.ndarray:
    return np.array([AA_TO_IDX.get(aa, 0) for aa in sequence], dtype=np.int64)


def process_dataset(raw_data: list, save_path: str = None) -> list:
    #compute distance matrices for all proteins
    processed = []
    for entry in tqdm(raw_data, desc="Computing distance matrices"):
        tokens = encode_sequence(entry['sequence'])
        dist_matrix = compute_distance_matrix(entry['coords'])
        processed.append({
            'pdb_id': entry['pdb_id'],
            'sequence': entry['sequence'],
            'tokens': tokens,
            'distance_matrix': dist_matrix,
            'coords': entry['coords'],
            'length': len(entry['sequence']),
        })

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(processed, f)
        print(f"Saved {len(processed)} processed proteins to {save_path}")

    return processed


def split_dataset(data: list, config: DataConfig, seed: int = 42):
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(data))

    n_train = int(len(data) * config.train_ratio)
    n_val = int(len(data) * config.val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train = [data[i] for i in train_idx]
    val = [data[i] for i in val_idx]
    test = [data[i] for i in test_idx]

    print(f"Split: {len(train)} train, {len(val)} val, {len(test)} test")
    return train, val, test


def generate_synthetic_data(
    num_proteins: int = 200,
    min_len: int = 30,
    max_len: int = 100,
    seed: int = 42,
) -> list:
    #generate fake protein data for testing the pipeline
    rng = np.random.RandomState(seed)
    aa_chars = list("ACDEFGHIKLMNPQRSTVWY")
    dataset = []

    for i in range(num_proteins):
        seq_len = rng.randint(min_len, max_len + 1)
        sequence = ''.join(rng.choice(aa_chars, size=seq_len))

        #coords along a noisy helix
        t = np.linspace(0, seq_len * 0.5, seq_len)
        helix_fraction = rng.uniform(0.3, 1.0)
        coords = np.zeros((seq_len, 3), dtype=np.float32)

        coords[:, 0] = 2.3 * np.cos(t * 1.75) * helix_fraction
        coords[:, 1] = 2.3 * np.sin(t * 1.75) * helix_fraction
        coords[:, 2] = 1.5 * t

        coords += rng.randn(seq_len, 3).astype(np.float32) * (1.0 - helix_fraction + 0.3)

        #enforce ~3.8A ca-ca distance
        for j in range(1, seq_len):
            vec = coords[j] - coords[j - 1]
            dist = np.linalg.norm(vec)
            if dist > 0:
                coords[j] = coords[j - 1] + vec / dist * (3.8 + rng.randn() * 0.1)

        dataset.append({
            'pdb_id': f'SYN_{i:04d}',
            'sequence': sequence,
            'coords': coords,
        })

    print(f"Generated {num_proteins} synthetic proteins (lengths {min_len}-{max_len}).")
    return dataset


class ProteinDataset(Dataset):
    def __init__(self, data: list, max_len: int = 200):
        self.data = data
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        L = entry['length']

        tokens = np.zeros(self.max_len, dtype=np.int64)
        tokens[:L] = entry['tokens'][:L]

        dist_matrix = np.zeros((self.max_len, self.max_len), dtype=np.float32)
        dist_matrix[:L, :L] = entry['distance_matrix'][:L, :L]

        mask = np.zeros((self.max_len, self.max_len), dtype=np.float32)
        mask[:L, :L] = 1.0

        coords = np.zeros((self.max_len, 3), dtype=np.float32)
        coords[:L] = entry['coords'][:L]

        return {
            'tokens': torch.from_numpy(tokens),
            'distance_matrix': torch.from_numpy(dist_matrix),
            'mask': torch.from_numpy(mask),
            'length': L,
            'coords': torch.from_numpy(coords),
        }


def get_dataloader(data: list, max_len: int, batch_size: int,
                   shuffle: bool = True, num_workers: int = 0) -> DataLoader:
    dataset = ProteinDataset(data, max_len=max_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


def prepare_data(config: DataConfig, use_synthetic: bool = False, seed: int = 42):
    #full pipeline: download or generate, process, split
    cache_path = os.path.join(config.processed_dir, "processed_proteins.pkl")

    if os.path.exists(cache_path) and not use_synthetic:
        print(f"Loading cached data from {cache_path}...")
        with open(cache_path, 'rb') as f:
            processed = pickle.load(f)
    else:
        if use_synthetic:
            raw = generate_synthetic_data(
                num_proteins=config.num_proteins,
                min_len=config.min_seq_length,
                max_len=min(config.max_seq_length, 100),
                seed=seed,
            )
        else:
            pdb_ids = search_pdb(config)
            raw = download_and_parse(pdb_ids, config)

        processed = process_dataset(raw, save_path=cache_path)

    train, val, test = split_dataset(processed, config, seed=seed)
    return train, val, test


if __name__ == "__main__":
    config = DataConfig(num_proteins=50)
    train, val, test = prepare_data(config, use_synthetic=True)

    loader = get_dataloader(train, max_len=config.max_seq_length, batch_size=4)
    batch = next(iter(loader))
    print(f"\nBatch shapes:")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape} ({v.dtype})")
        else:
            print(f"  {k}: {v}")