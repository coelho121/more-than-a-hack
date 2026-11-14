"""Frente 1: quando é que a bola está morta (D).

Estado atual (passo 1): só o período antes do pontapé de saída de cada parte.

Como se encontra o pontapé de saída, sem nenhum frame escrito à mão:
  1. o Smart Tagging tem um SET PIECES do tipo "Kick Off (Start)" por parte, que marca um
     intervalo de alguns segundos à volta do reinício;
  2. dentro desse intervalo, o pontapé de saída é o momento em que a bola deixa o centro do
     campo (último frame em que o tracking a vê a menos de CENTRE_RADIUS_M do centro);
  3. se a bola nunca for vista no centro nesse intervalo, usa-se o meio do intervalo.

Tudo o que vai do início da parte até esse momento é D.

Próximos passos (3 e 4): paragens a partir dos set pieces e da geometria.

Contrato: dead_ball_mask(atd, tags) devolve um array bool (N,) com True onde achamos que a bola
está morta, ou None se não houver Smart Tagging.
"""

import numpy as np

from src import features, match

KICKOFF_CODE = "SET PIECES"
KICKOFF_TYPE = "kick off (start)"

# A bola está "no centro" se estiver a menos disto do ponto central. A marca do centro é um
# ponto, mas o tracking automático tem ruído de algumas dezenas de centímetros; 1.5 m dá folga
# e a bola percorre essa distância em 3 ou 4 frames depois do toque.
CENTRE_RADIUS_M = 1.5


def kickoff_frame(atd, start, end):
    """(frame, método) do pontapé de saída dentro do intervalo [start, end) do tagging."""
    n_frames = len(atd["ball_xy"])
    start, end = max(0, int(start)), min(n_frames, int(end))
    if end <= start:
        return None, "intervalo vazio"

    ball = features.to_metres(atd["ball_xy"][start:end], atd["pitch"])
    centre = np.asarray(atd["pitch"], dtype=np.float64) / 2.0
    dist = np.hypot(ball[:, 0] - centre[0], ball[:, 1] - centre[1])  # NaN sem bola
    inside = dist <= CENTRE_RADIUS_M
    outside = dist > CENTRE_RADIUS_M

    if not inside.any():
        return (start + end) // 2, "meio do intervalo (bola nunca vista no centro)"

    first_inside = int(np.argmax(inside))
    exits = np.flatnonzero(outside[first_inside:])
    if len(exits):
        first_exit = first_inside + int(exits[0])
        last_inside = int(np.flatnonzero(inside[:first_exit])[-1])
    else:
        last_inside = int(np.flatnonzero(inside)[-1])
    return start + last_inside + 1, "bola a sair do centro"


def kickoffs(atd, tags):
    """[{half, first, frame, method}] um por parte em que o tagging tem pontapé de saída."""
    out = []
    if tags is None or not len(tags):
        return out
    types = tags["type"].fillna("").str.strip().str.lower()
    is_kickoff = (tags["code"] == KICKOFF_CODE) & (types == KICKOFF_TYPE)

    for half, first, last in match.half_blocks():
        overlaps = is_kickoff & (tags["end_frame"] > first) & (tags["start_frame"] <= last)
        candidates = tags[overlaps]
        if not len(candidates):
            continue
        row = candidates.iloc[0]  # a tabela vem ordenada por start_frame
        frame, method = kickoff_frame(atd, row["start_frame"], row["end_frame"])
        if frame is None:
            continue
        frame = min(max(int(frame), first), last)
        out.append({"half": half, "first": first, "frame": frame, "method": method})
    return out


def dead_ball_mask(atd, tags=None):
    if tags is None:
        return None
    mask = np.zeros(len(atd["ball_xy"]), dtype=bool)
    for kickoff in kickoffs(atd, tags):
        mask[kickoff["first"]:kickoff["frame"]] = True
    return mask