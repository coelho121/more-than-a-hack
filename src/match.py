"""Estrutura do jogo: que frames pertencem a cada parte e quais são avaliados.

Os limites das partes vêm do games.py oficial (half_video_blocks), não estão escritos aqui.
"""

import numpy as np

from games import GAME, GAMES, half_video_blocks


def half_blocks(game=GAME):
    """[(parte, primeiro_frame, ultimo_frame)] em frames do vídeo, limites incluídos."""
    return [(int(half), int(first), int(last)) for half, first, last in half_video_blocks(game)]


def half_runs(game=GAME):
    """[(inicio, fim)] meio-abertos, o formato que o scorer oficial usa."""
    return [(first, last + 1) for _, first, last in half_blocks(game)]


def graded_mask(n_frames=None, game=GAME):
    """(N,) bool: True nos frames das duas partes (os únicos que contam para o score)."""
    n_frames = n_frames or GAMES[game]["n_video_frames"]
    mask = np.zeros(n_frames, dtype=bool)
    for start, stop in half_runs(game):
        mask[start:stop] = True
    return mask