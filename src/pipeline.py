"""Junta as frentes e devolve uma label por frame do vídeo tático.

    labels = pipeline.run(atd, tags)        # array de 'H'/'A'/'D'/'X', tamanho 199319
    labels = pipeline.baseline(atd, 50)     # a baseline oficial, só para o teste de sanidade

Cada versão nova do método é uma mudança aqui (ou nos módulos que isto chama), nunca no notebook.
Assim o make_submission.py gera sempre o ficheiro a partir do mesmo sítio.
"""

import numpy as np

from src import dead_ball, match, possession


def baseline(atd, window=50):
    """Exatamente a baseline oficial: jogador mais perto da bola + voto por maioria."""
    codes = possession.nearest_player_codes(atd)
    return possession.majority_smooth(codes, window)


def run(atd, tags=None, window=50):
    labels = baseline(atd, window)

    dead = dead_ball.dead_ball_mask(atd, tags)
    if dead is not None:
        labels = np.where(dead, "D", labels)

    # Fora das duas partes não há nada para avaliar: X.
    labels = np.where(match.graded_mask(len(labels)), labels, "X")
    return labels