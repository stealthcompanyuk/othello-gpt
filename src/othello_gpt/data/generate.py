import multiprocessing as mp
from multiprocessing import Pool
from typing import Dict, List

import numpy as np
from datasets import Dataset, concatenate_datasets
from tqdm import tqdm

from othello_gpt.othello import OthelloState, get_legal_move_ids, is_terminal, make_move


def generate_game(size: int, no_pass: bool = True) -> Dict[str, List]:
    legalities = []
    moves = []
    boards = []

    state = OthelloState(size)

    while not is_terminal(state):
        legal_move_ids = get_legal_move_ids(state, no_pass=no_pass)

        if not legal_move_ids:
            return generate_game(size, no_pass=no_pass)

        legalities.append(np.zeros((size, size), dtype=bool))
        for move_id in legal_move_ids:
            if move_id != size * size:
                legalities[-1][*divmod(move_id, size)] = 1

        move_id = np.random.choice(legal_move_ids)
        moves.append(move_id)

        state = make_move(state, move_id, validate=False)
        boards.append(state.board)

    return {
        "legalities": legalities,
        "moves": moves,
        "boards": boards,
    }


def generate_dataset(
    n_games: int,
    size: int,
    batch_size: int = 10000,
    no_pass: bool = True,
    batch_id: int = 0,
) -> Dataset:
    if n_games > batch_size:
        batch_sizes = [batch_size] * (n_games // batch_size)
        if n_games % batch_size > 0:
            batch_sizes.append(n_games % batch_size)

        args = [(n, size, batch_size, no_pass, i) for i, n in enumerate(batch_sizes)]
        with Pool(mp.cpu_count()) as pool:
            datasets = list(pool.starmap(generate_dataset, tqdm(args, position=0)))
        return concatenate_datasets(datasets)

    pbar = tqdm(range(n_games), desc=f"{batch_id=}", position=batch_id + 1, leave=None)
    games = [generate_game(size, no_pass=no_pass) for _ in pbar]
    games = {k: [game[k] for game in games] for k in games[0].keys()}

    dataset = Dataset.from_dict(games)
    return dataset


if __name__ == "__main__":
    from pathlib import Path

    root_dir = Path().cwd()
    data_dir = root_dir / "data"
    data_dir.mkdir(exist_ok=True)

    n_games = 1000000
    size = 6

    dataset_dict_path = data_dir / f"othello_{n_games}_{size}"

    dataset = generate_dataset(n_games, size)
    dataset_dict = dataset.train_test_split(test_size=0.1)
    dataset_dict.save_to_disk(dataset_dict_path)
