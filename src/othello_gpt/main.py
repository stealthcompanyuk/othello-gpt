from pathlib import Path

import numpy as np
from datasets import Dataset

from othello_gpt.data.generate import generate_dataset
from othello_gpt.data.vis import plot_game

if __name__ == "__main__":
    root_dir = Path.cwd()
    data_dir = root_dir / "data"
    data_dir.mkdir(exist_ok=True)

    n_games = 10000
    size = 6

    keys = ["boards", "legal_moves", "histories"]
    npy_paths = [data_dir / f"othello_{n_games}_{size}_{k}.npy" for k in keys]
    if all(p.exists() for p in npy_paths):
        dataset = {k: np.load(p) for k, p in zip(keys, npy_paths)}
    else:
        dataset = generate_dataset(n_games, size)
        for k, v in dataset.items():
            np.save(data_dir / f"othello_{n_games}_{size}_{k}.npy", v)

    dataset = Dataset.from_dict(dataset)
    plot_game(dataset[0], subplot_size=180, n_cols=6)
