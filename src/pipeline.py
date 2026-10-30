"""Junta as frentes e devolve uma label por frame do vídeo tático.

    labels = pipeline.run(atd, tags, window=50)   # array de 'H'/'A'/'D'/'X', tamanho 199319

Cada versão nova do método é uma mudança aqui (ou nos módulos que isto chama), nunca no notebook.
Assim o make_submission.py gera sempre o ficheiro a partir do mesmo sítio.
"""

import numpy as np

from src import dead_ball, possession


def run(atd, tags=None, window=50):
    codes = possession.nearest_player_codes(atd)
    labels = possession.majority_smooth(codes, window)

    dead = dead_ball.dead_ball_mask(atd, tags)
    if dead is not None:
        labels = np.where(dead, "D", labels)

    return labels
