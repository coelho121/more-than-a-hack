"""Coisas calculadas a partir do tracking. Tudo vetorizado sobre os frames.

Regra: as coordenadas do ATD são 0-1 e o campo não é quadrado, por isso qualquer distância é
calculada em METROS (to_metres primeiro). Sem isso, as distâncias no eixo curto ficam esticadas.
"""

import numpy as np


def to_metres(xy, pitch):
    """(..., 2) em 0-1 -> metros."""
    return np.asarray(xy, dtype=np.float64) * np.asarray(pitch, dtype=np.float64)


def ball_player_sq_distances(atd):
    """(N, S) distância AO QUADRADO em metros². Comparar quadrados evita a raiz e é exatamente
    o que o baseline oficial faz (importa para reproduzir o ficheiro dele bit a bit)."""
    ball = to_metres(atd["ball_xy"], atd["pitch"])[:, None, :]
    players = to_metres(atd["players_xy"], atd["pitch"])
    dx = players[..., 0] - ball[..., 0]
    dy = players[..., 1] - ball[..., 1]
    return dx * dx + dy * dy


def ball_player_distances(atd):
    """(N, S) distância em metros de cada slot à bola. NaN onde falta a bola ou o jogador."""
    return np.sqrt(ball_player_sq_distances(atd))


def nearest_player(atd):
    """(dist, team) do slot mais perto da bola em cada frame.

    team: 0 casa, 1 fora, -1 quando não há resposta (sem bola ou sem jogadores).
    Em empate fica o primeiro slot, tal como no baseline oficial.
    """
    sq = ball_player_sq_distances(atd)
    has_answer = np.isfinite(sq).any(axis=1)
    safe = np.where(np.isfinite(sq), sq, np.inf)
    slot = np.argmin(safe, axis=1)

    team = np.full(len(sq), -1, dtype=np.int8)
    team[has_answer] = atd["team_of_track"][slot[has_answer]]
    nearest = np.full(len(sq), np.nan, dtype=np.float64)
    nearest[has_answer] = np.sqrt(safe[has_answer, slot[has_answer]])
    return nearest, team


def ball_visible(atd):
    """(N,) bool: o tracking vê a bola neste frame."""
    return np.isfinite(atd["ball_xy"]).all(axis=1)
