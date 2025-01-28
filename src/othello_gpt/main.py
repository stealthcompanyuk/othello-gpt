from pathlib import Path

import numpy as np
from datasets import Dataset, DatasetDict

from othello_gpt.data.generate import generate_dataset
from othello_gpt.data.vis import plot_game

if __name__ == "__main__":
    root_dir = Path.cwd()
    data_dir = root_dir / "data"
    data_dir.mkdir(exist_ok=True)

    n_games = 100000
    size = 6

    dataset_dict_path = data_dir / f"othello_{n_games}_{size}"

    if dataset_dict_path.exists():
        dataset_dict = DatasetDict.load_from_disk(dataset_dict_path)
    else:
        dataset_dict = generate_dataset(n_games, size)
        dataset_dict.save_to_disk(dataset_dict_path)

    plot_game(dataset_dict["test"][0], subplot_size=180, n_cols=4)
