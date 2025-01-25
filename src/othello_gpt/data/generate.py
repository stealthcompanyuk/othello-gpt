import numpy as np
from tqdm import tqdm
from typing import Dict

from othello_gpt.othello import get_max_moves, OthelloState, get_legal_move_ids, make_move, is_terminal


def generate_dataset(n_games: int, size: int) -> Dict[str, np.ndarray]:
    max_moves = get_max_moves(size)

    boards = np.zeros((n_games, max_moves, size, size))
    legal_moves = np.zeros((n_games, max_moves, size, size))
    histories = -np.ones((n_games, max_moves))

    for i in tqdm(range(n_games)):
        state = OthelloState(size)
        for j in range(max_moves):
            if is_terminal(state):
                break

            legal_move_ids = get_legal_move_ids(state)
            for move_id in legal_move_ids:
                if move_id != size * size:
                    legal_moves[i, j, *divmod(move_id, size)] = 1

            move_id = np.random.choice(legal_move_ids)
            histories[i, j] = move_id

            state = make_move(state, move_id, validate=False)
            boards[i, j] = state.board

    return {
        "boards": boards,
        "legal_moves": legal_moves,
        "histories": histories,
    }
