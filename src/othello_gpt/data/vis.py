from typing import Dict, List

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def move_id_to_coord(move_id: int, size: int) -> tuple:
    return tuple(divmod(move_id, size))


def move_id_to_text(move_id: int, size: int) -> str:
    if move_id == -1:
        return "PAD"
    if move_id == size * size:
        return "PASS"
    y, x = move_id_to_coord(move_id, size)
    return f"{chr(ord('A') + x)}{y + 1}"


def plot_game(game: Dict[str, List], subplot_size=180, n_cols=8):
    game_boards = np.array(game["boards"])
    game_legal_moves = np.array(game["legal_moves"])
    game_history = np.array(game["histories"])
    size = game_boards.shape[1]

    row_labels = list(map(str, range(1, 1 + size)))
    col_labels = [chr(ord("A") + i) for i in range(size)]
    hovertext = np.array([[f"{col}{row}" for col in col_labels] for row in row_labels])
    margin = subplot_size // 8

    n_moves = (game_history != -1).sum().item()
    n_rows = (n_moves - 1) // n_cols + 1
    subplot_titles = [
        f"{i + 1}. {move_id_to_text(int(move_id), size)}"
        for i, move_id in enumerate(game_history[:n_moves])
    ]

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        shared_xaxes=False,
        shared_yaxes=True,
        vertical_spacing=0.1,
    )

    for i in range(n_moves):
        row = i // n_cols + 1
        col = i % n_cols + 1

        # Create str 2d array for legal moves where 0 -> "" and 1 -> "X"
        if i + 1 < n_moves:
            text = np.where(game_legal_moves[i + 1], "X", "")
        else:
            text = np.full_like(game_legal_moves[0], "", dtype=str)
        text[*move_id_to_coord(int(game_history[i]), size)] = "o"

        fig.add_trace(
            go.Heatmap(
                z=game_boards[i],
                colorscale="gray_r",
                showscale=False,
                text=text,
                hovertext=hovertext,
                hovertemplate="%{hovertext}<extra></extra>",
                x=col_labels,
                y=row_labels,
                xgap=0.2,
                ygap=0.2,
                texttemplate="%{text}",
                textfont={"color": "black" if i % 2 else "white"},
            ),
            row=row,
            col=col,
        )

    fig.update_yaxes(
        showline=True,
        linecolor="black",
        linewidth=1,
        mirror=True,
        constrain="domain",
        autorange="reversed",
    )

    fig.update_xaxes(
        showline=True,
        linecolor="black",
        linewidth=1,
        mirror=True,
        scaleanchor="y",
        scaleratio=1,
        constrain="domain",
    )

    fig.update_layout(
        font=dict(size=subplot_size // 20),
        margin=dict(l=margin, r=margin, t=margin, b=margin),
        width=subplot_size * n_cols,
        height=subplot_size * n_rows,
    )

    fig.update_annotations(font_size=subplot_size // 10)

    fig.show()
