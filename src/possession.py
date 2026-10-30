"""Frente 2: quem tem a bola (H/A) e quando é que a posse muda.

Estado atual (v0): igual ao baseline oficial. Jogador mais perto da bola, depois voto por
maioria numa janela centrada. Serve para confirmar que o pipeline está certo: com janela 50
tem de dar EXATAMENTE o mesmo ficheiro que evaluation/results/baseline_smooth50.csv.

Ideias (ver docs/oficial/CHALLENGE.md e EVALUATION.md):
  - a posse só muda com CONTROLO: um toque, um desvio ou um duelo aéreo não chegam;
  - em vez de janela, algo com memória (histerese, HMM/Viterbi com custo por troca);
  - o spellF1 castiga partir posses ao meio: trocas falsas custam mais do que atrasos pequenos;
  - X no meio de uma posse é grátis no spellF1, numa transição é perigoso.
"""

import numpy as np

from src import features

NOBODY = -1


def nearest_player_codes(atd):
    """(N,) 0 casa, 1 fora, -1 sem resposta. Só há resposta onde a bola é visível."""
    _, team = features.nearest_player(atd)
    team = team.copy()
    team[~features.ball_visible(atd)] = NOBODY
    return team


def majority_smooth(codes, window):
    """Voto por maioria numa janela de +/- window frames. Devolve 'H'/'A'/'D'.

    Mesma lógica que starter/baseline.py: D onde ninguém tem a bola na janela inteira,
    empates vão para a casa.
    """
    if window <= 0:
        return np.where(codes == 0, "H", np.where(codes == 1, "A", "D"))

    home_cum = np.concatenate(([0], np.cumsum(codes == 0)))
    away_cum = np.concatenate(([0], np.cumsum(codes == 1)))
    index = np.arange(len(codes))
    low = np.maximum(0, index - window)
    high = np.minimum(len(codes), index + window + 1)
    home = home_cum[high] - home_cum[low]
    away = away_cum[high] - away_cum[low]

    labels = np.full(len(codes), "D", dtype="<U1")
    seen = (home + away) > 0
    labels[seen] = np.where(home[seen] >= away[seen], "H", "A")
    return labels
