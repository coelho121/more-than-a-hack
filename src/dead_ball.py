"""Frente 1: quando é que a bola está morta (D).

Estado atual (passo 2). A bola é dada como morta em três situações:

  1. ANTES DO PONTAPÉ DE SAÍDA de cada parte. O Smart Tagging tem um SET PIECES do tipo
     "Kick Off (Start)" por parte, que marca um intervalo de alguns segundos à volta do
     reinício; dentro dele, o pontapé de saída é o momento em que a bola deixa o centro.

  2. BOLA VISTA FORA DO CAMPO, mais de OUT_MARGIN_M para lá de uma linha.

  3. BURACOS LONGOS SEM BOLA em que o contexto diz que ela saiu. O tracking perde a bola em
     duas situações muito diferentes: quando ela sai do campo (bola morta) e quando fica
     tapada com o jogo a decorrer (por exemplo nas mãos do guarda-redes). Por isso um buraco
     só é D se:
       - a bola foi vista pela última vez junto a uma linha limite, ou
       - ia a caminho de sair (posição + velocidade), ou
       - reaparece junto a uma linha limite (lançamento, canto).
     Caso contrário o jogo continua e a posse fica com quem a tinha (src/possession.py).

No fim há uma limpeza com lógica de futebol: uma paragem não dura menos de MIN_DEAD_S e não
há jogo vivo durante menos de MIN_ALIVE_S entre duas paragens.

Ainda por fazer (passo 3): bola visível mas parada antes de um reinício (livres, pontapés de
baliza, cantos) e os golos.

Contrato: dead_ball_mask(atd, tags) devolve um array bool (N,), True onde a bola está morta.
"""

import numpy as np

from src import features, match

FPS = features.FPS

KICKOFF_CODE = "SET PIECES"
KICKOFF_TYPE = "kick off (start)"

# A bola está "no centro" se estiver a menos disto do ponto central. O tracking automático tem
# ruído de algumas dezenas de centímetros; 1.5 m dá folga e a bola percorre essa distância em
# 3 ou 4 frames depois do toque.
CENTRE_RADIUS_M = 1.5

# Bola fora: mais de meio metro para lá da linha. A bola só está fora quando passa a linha
# toda, e o meio metro absorve o ruído do tracking.
OUT_MARGIN_M = 0.5

# Um buraco só conta como possível paragem a partir de 2 s. O valor é pouco sensível: no jogo
# inteiro quase não há buracos entre 0.5 s e 5 s (ver scripts/report.py).
MIN_GAP_S = 2.0

# "Junto a uma linha": a menos de 2 m. Um jogador com a bola controlada raramente está tão
# perto da linha, e uma bola rápida anda quase 1 m por frame, por isso o último frame visto
# antes de sair fica tipicamente dentro desta faixa.
EDGE_M = 2.0

# "A caminho de sair": mantendo a velocidade dos últimos frames, a bola estaria fora do campo
# daqui a 1 s.
LOOKAHEAD_S = 1.0
VELOCITY_FRAMES = 5

# Limpeza. O reinício mais rápido (um lançamento apressado) demora uns segundos, e não há
# jogadas de menos de 2 s entre duas paragens.
MIN_DEAD_S = 2.0
MIN_ALIVE_S = 2.0


# -- pontapé de saída ---------------------------------------------------------------------------

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


def prekickoff_mask(atd, tags):
    mask = np.zeros(len(atd["ball_xy"]), dtype=bool)
    for kickoff in kickoffs(atd, tags):
        mask[kickoff["first"]:kickoff["frame"]] = True
    return mask


# -- bola fora do campo -------------------------------------------------------------------------

def outside_mask(atd):
    """(N,) bool: a bola vê-se e está mais de OUT_MARGIN_M para lá de uma linha."""
    edge = features.edge_distance(features.ball_metres(atd), atd["pitch"])
    return np.isfinite(edge) & (edge < -OUT_MARGIN_M)


# -- buracos sem bola ---------------------------------------------------------------------------

def _velocity(ball_m, visible, frame, lowest):
    """Velocidade (m/s) da bola no frame, a partir dos últimos VELOCITY_FRAMES frames vistos."""
    first = frame
    while first - 1 >= lowest and frame - (first - 1) <= VELOCITY_FRAMES and visible[first - 1]:
        first -= 1
    if first == frame:
        return np.zeros(2)
    return (ball_m[frame] - ball_m[first]) / float(frame - first) * FPS


def classify_gap(ball_m, visible, pitch, start, stop, half_start, half_stop):
    """(está_morta, razão) para um buraco [start, stop) dentro de uma parte."""
    if start <= half_start:
        return True, "início da parte"
    if stop >= half_stop:
        return True, "fim da parte"

    before = start - 1                      # último frame com bola antes do buraco
    edge_before = float(features.edge_distance(ball_m[before], pitch))
    if edge_before <= EDGE_M:
        return True, "vista pela última vez junto a uma linha"

    velocity = _velocity(ball_m, visible, before, half_start)
    ahead = ball_m[before] + velocity * LOOKAHEAD_S
    if float(features.edge_distance(ahead, pitch)) < 0.0:
        return True, "ia a caminho de sair"

    edge_after = float(features.edge_distance(ball_m[stop], pitch))  # 1.º frame com bola depois
    if edge_after <= EDGE_M:
        return True, "reaparece junto a uma linha"

    return False, "jogo continua (bola tapada)"


def gap_table(atd):
    """[{start, stop, dead, reason}] todos os buracos de pelo menos MIN_GAP_S, por ordem."""
    visible = features.ball_visible(atd)
    ball_m = features.ball_metres(atd)
    min_frames = int(round(MIN_GAP_S * FPS))
    out = []
    for half_start, half_stop in match.half_runs():
        starts, stops = features.runs_of(~visible[half_start:half_stop])
        for a, b in zip(starts.tolist(), stops.tolist()):
            if b - a < min_frames:
                continue
            start, stop = half_start + a, half_start + b
            dead, reason = classify_gap(ball_m, visible, atd["pitch"], start, stop,
                                        half_start, half_stop)
            out.append({"start": start, "stop": stop, "dead": dead, "reason": reason})
    return out


def gap_mask(atd):
    mask = np.zeros(len(atd["ball_xy"]), dtype=bool)
    for gap in gap_table(atd):
        if gap["dead"]:
            mask[gap["start"]:gap["stop"]] = True
    return mask


# -- limpeza ------------------------------------------------------------------------------------

def clean(mask):
    """Tira paragens de menos de MIN_DEAD_S e fecha jogo vivo de menos de MIN_ALIVE_S entre
    duas paragens. Trabalha parte a parte; os extremos de cada parte não são tocados."""
    mask = mask.copy()
    min_dead = int(round(MIN_DEAD_S * FPS))
    min_alive = int(round(MIN_ALIVE_S * FPS))
    for half_start, half_stop in match.half_runs():
        window = mask[half_start:half_stop]
        n = len(window)

        starts, stops = features.runs_of(window)
        for a, b in zip(starts.tolist(), stops.tolist()):
            if b - a < min_dead and a > 0 and b < n:
                window[a:b] = False

        starts, stops = features.runs_of(~window)
        for a, b in zip(starts.tolist(), stops.tolist()):
            if b - a < min_alive and a > 0 and b < n:
                window[a:b] = True
    return mask


# -- tudo junto ---------------------------------------------------------------------------------

def components(atd, tags=None):
    """As peças separadas, para o relatório: {prekickoff, outside, gaps, final}."""
    prekickoff = prekickoff_mask(atd, tags)
    outside = outside_mask(atd)
    gaps = gap_mask(atd)
    final = clean(outside | gaps) | prekickoff
    return {"prekickoff": prekickoff, "outside": outside, "gaps": gaps, "final": final}


def dead_ball_mask(atd, tags=None):
    return components(atd, tags)["final"]