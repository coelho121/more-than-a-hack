"""Coisas calculadas a partir do tracking. Tudo vetorizado sobre os frames.

Regra: as coordenadas do ATD são 0-1 e o campo não é quadrado, por isso qualquer distância é
calculada em METROS (to_metres primeiro). Sem isso, as distâncias no eixo curto ficam esticadas.
"""

import numpy as np

FPS = 25


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


def team_nearest_distances(atd):
    """(d_casa, d_fora): distância em metros do jogador mais perto de CADA equipa à bola.

    Infinito onde não há resposta (sem bola, ou nenhum jogador dessa equipa seguido), para as
    comparações "<=" darem sempre False nesses frames.
    """
    sq = ball_player_sq_distances(atd)
    safe = np.where(np.isfinite(sq), sq, np.inf)
    teams = np.asarray(atd["team_of_track"])
    home = np.sqrt(safe[:, teams == 0].min(axis=1))
    away = np.sqrt(safe[:, teams == 1].min(axis=1))
    return home, away


def ball_visible(atd):
    """(N,) bool: o tracking vê a bola neste frame."""
    return np.isfinite(atd["ball_xy"]).all(axis=1)


def ball_metres(atd):
    """(N, 2) posição da bola em metros. NaN quando não se vê."""
    return to_metres(atd["ball_xy"], atd["pitch"])


def edge_distance(ball_m, pitch):
    """(N,) distância em metros da bola à linha limite mais próxima.

    Positiva dentro do campo, negativa fora, NaN sem bola.
    """
    length, width = float(pitch[0]), float(pitch[1])
    x, y = ball_m[..., 0], ball_m[..., 1]
    return np.minimum(np.minimum(x, length - x), np.minimum(y, width - y))


def runs_of(mask):
    """(inícios, fins) meio-abertos dos troços consecutivos de True num array bool."""
    edges = np.diff(np.concatenate(([0], np.asarray(mask).astype(np.int8), [0])))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)