import numpy as np
from tqdm import tqdm
from typing import List, Tuple
from datasets import Dataset, concatenate_datasets

from othello_gpt.othello import OthelloState, get_legal_move_ids, make_move, is_terminal


def generate_game(size: int, no_pass: bool = True) -> Tuple[List[np.ndarray], List[int], List[np.ndarray]]:
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

    return legalities, moves, boards


def generate_dataset(n_games: int, size: int, batch_size: int = 10000, no_pass: bool = True):
    datasets = []

    legalities = []
    histories = []
    boards = []

    for i in tqdm(range(n_games)):
        l, m, b = generate_game(size, no_pass=no_pass)
        legalities.append(l)
        histories.append(m)
        boards.append(b)

        if i % batch_size == 0 and i > 0:
            partial_dataset = Dataset.from_dict({
                "legalities": legalities,
                "histories": histories,
                "boards": boards,
            })
            datasets.append(partial_dataset)
            legalities = []
            histories = []
            boards = []

    partial_dataset = Dataset.from_dict({
        "legalities": legalities,
        "histories": histories,
        "boards": boards,
    })
    datasets.append(partial_dataset)

    dataset = concatenate_datasets(datasets)
    return dataset
