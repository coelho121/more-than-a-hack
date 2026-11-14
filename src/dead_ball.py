"""Frente 1: quando é que a bola está morta (D).

Estado atual (passo 3). A bola é dada como morta em cinco situações:

  1. ANTES DO PONTAPÉ DE SAÍDA de cada parte. O Smart Tagging tem um SET PIECES do tipo
     "Kick Off (Start)" por parte, que marca um intervalo de alguns segundos à volta do
     reinício; dentro dele, o pontapé de saída é o momento em que a bola deixa o centro.

  2. BOLA VISTA FORA DO CAMPO, mais de OUT_MARGIN_M para lá de uma linha.

  3. BURACOS SEM BOLA (períodos seguidos em que o tracking não tem posição da bola). A bola
     desaparece do tracking quando sai do campo, quando alguém a apanha com as mãos, ou
     quando fica tapada. As perguntas são feitas por esta ordem, e a primeira que der "sim"
     decide:
       a) o buraco apanha o início ou o fim da parte?                               -> D
       b) a bola foi vista pela última vez junto a uma linha limite?                -> D
       c) ia a caminho de sair (posição + velocidade)?                              -> D
       d) reaparece junto a uma linha limite?                                       -> D
       e) dura mais de MAX_HIDDEN_S?  (ninguém esconde a bola tanto tempo em jogo)  -> D
       f) sumiu e voltou fora das grandes áreas?  (fora da área só o árbitro parado
          explica uma bola na mão ou desaparecida tanto tempo)                      -> D
       g) sobra: sumiu ou voltou dentro de uma grande área e durou pouco. É o caso
          do guarda-redes com a bola nas mãos ou de um molho de jogadores na área.  -> vivo

  4. BOLA PARADA À VISTA. A bola quase não sai do sítio (REST_RANGE_M) durante pelo menos 1 s.
     Isso é bola morta se:
       a) acontece colado a outra paragem (a bola voltou e está à espera do reinício), ou
       b) dura pelo menos MIN_REST_S e a bola está sozinha (ninguém a menos de
          UNATTENDED_M) durante pelo menos 1 s: é a bola colocada para um livre, canto
          ou pontapé de baliza, com o marcador a tomar balanço, ou
       c) dura pelo menos LONG_REST_S, com ou sem jogador em cima.
     A paragem acaba quando a bola sai do sítio onde estava (o pontapé ou o lançamento).

  5. GOLOS. Para cada "Kick Off (After Goal)" do Smart Tagging, tudo é D desde o golo até ao
     pontapé de saída. O golo é a última vez que a bola foi vista na zona da baliza; se nunca
     foi vista lá, usa-se a última paragem já detetada; em último caso, GOAL_DEFAULT_S antes.

No fim há uma limpeza com lógica de futebol: uma paragem não dura menos de MIN_DEAD_S e não
há jogo vivo durante menos de MIN_ALIVE_S entre duas paragens.

Contrato: dead_ball_mask(atd, tags) devolve um array bool (N,), True onde a bola está morta.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from src import features, match

FPS = features.FPS

SET_PIECE_CODE = "SET PIECES"
KICKOFF_TYPE = "kick off (start)"
GOAL_KICKOFF_TYPE = "kick off (after goal)"

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

# Com o jogo a decorrer, o caso em que a bola fica mais tempo escondida é o guarda-redes com
# ela nas mãos. A lei dá-lhe poucos segundos e na prática raramente passa de 10 a 15.
MAX_HIDDEN_S = 15.0

# Grande área: 16.5 m de profundidade, 40.32 m de largura (20.16 m para cada lado do centro).
BOX_DEPTH_M = 16.5
BOX_HALF_WIDTH_M = 20.16

# Bola parada: num segundo inteiro não se afasta mais de 1 m (em x e em y). O ruído do
# tracking numa bola parada fica bem abaixo disso, e um jogador a conduzir a bola, mesmo
# devagar, anda mais de 1 m num segundo.
REST_WINDOW_FRAMES = 25
REST_RANGE_M = 1.0
# Bola "sozinha": nenhum jogador a menos de 2 m (o mesmo raio usado para o controlo).
UNATTENDED_M = 2.0
UNATTENDED_MIN_S = 1.0
# Bola parada e sozinha durante 3 s: em jogo corrido ninguém larga a bola tanto tempo.
MIN_REST_S = 3.0
# Bola parada 8 s, mesmo com um jogador em cima: em jogo corrido não acontece.
LONG_REST_S = 8.0
# Uma bola parada está "colada" a outra paragem se começa ou acaba a menos de 2 s dela.
ADJACENT_S = 2.0

# Golos. Meia baliza são 3.66 m; a margem de 1 m absorve o ruído. Do golo ao pontapé de saída
# passa tipicamente cerca de um minuto; 120 s cobre também uma revisão curta do VAR.
GOAL_HALF_WIDTH_M = 3.66
GOAL_MARGIN_M = 1.0
GOAL_LOOKBACK_S = 120.0
GOAL_CLUSTER_S = 5.0
GOAL_DEFAULT_S = 40.0

# Limpeza. O reinício mais rápido (um lançamento apressado) demora uns segundos, e não há
# jogadas de menos de 2 s entre duas paragens.
MIN_DEAD_S = 2.0
MIN_ALIVE_S = 2.0


def _frames(seconds):
    return int(round(seconds * FPS))


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


def _set_pieces_of_type(tags, wanted):
    if tags is None or not len(tags):
        return None
    types = tags["type"].fillna("").str.strip().str.lower()
    return tags[(tags["code"] == SET_PIECE_CODE) & (types == wanted)]


def kickoffs(atd, tags):
    """[{half, first, frame, method}] um por parte em que o tagging tem pontapé de saída."""
    out = []
    rows = _set_pieces_of_type(tags, KICKOFF_TYPE)
    if rows is None:
        return out
    for half, first, last in match.half_blocks():
        candidates = rows[(rows["end_frame"] > first) & (rows["start_frame"] <= last)]
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

def in_penalty_area(position, pitch):
    """A posição (x, y) em metros está dentro de uma das duas grandes áreas?"""
    length, width = float(pitch[0]), float(pitch[1])
    x, y = float(position[0]), float(position[1])
    return ((x <= BOX_DEPTH_M or x >= length - BOX_DEPTH_M)
            and abs(y - width / 2.0) <= BOX_HALF_WIDTH_M)


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
        return True, "a) início da parte"
    if stop >= half_stop:
        return True, "a) fim da parte"

    before = start - 1                      # último frame com bola antes do buraco
    edge_before = float(features.edge_distance(ball_m[before], pitch))
    if edge_before <= EDGE_M:
        return True, "b) sumiu junto a uma linha"

    velocity = _velocity(ball_m, visible, before, half_start)
    ahead = ball_m[before] + velocity * LOOKAHEAD_S
    if float(features.edge_distance(ahead, pitch)) < 0.0:
        return True, "c) ia a caminho de sair"

    edge_after = float(features.edge_distance(ball_m[stop], pitch))  # 1.º frame com bola depois
    if edge_after <= EDGE_M:
        return True, "d) só a reaparição é junto a uma linha"

    if stop - start > _frames(MAX_HIDDEN_S):
        return True, "e) mais de {:.0f} s sem bola".format(MAX_HIDDEN_S)

    if not (in_penalty_area(ball_m[before], pitch) or in_penalty_area(ball_m[stop], pitch)):
        return True, "f) sumiu e voltou fora das áreas"

    return False, "g) curto e numa área (guarda-redes)"


def gap_table(atd):
    """[{start, stop, dead, reason}] todos os buracos de pelo menos MIN_GAP_S, por ordem."""
    visible = features.ball_visible(atd)
    ball_m = features.ball_metres(atd)
    out = []
    for half_start, half_stop in match.half_runs():
        starts, stops = features.runs_of(~visible[half_start:half_stop])
        for a, b in zip(starts.tolist(), stops.tolist()):
            if b - a < _frames(MIN_GAP_S):
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


# -- bola parada à vista ------------------------------------------------------------------------

def stationary_flags(atd):
    """(N,) bool: a bola vê-se e, no segundo centrado neste frame, não sai de um quadrado de
    REST_RANGE_M de lado. Exige bola visível em pelo menos metade desse segundo."""
    ball_m = features.ball_metres(atd)
    visible = features.ball_visible(atd)
    half = REST_WINDOW_FRAMES // 2

    def rolling(values, fill, reducer):
        padded = np.pad(np.where(visible, values, fill), half, constant_values=fill)
        return reducer(sliding_window_view(padded, 2 * half + 1), axis=1)

    seen = rolling(np.ones(len(visible)), 0.0, np.sum)
    enough = visible & (seen >= half + 1)
    flags = np.zeros(len(visible), dtype=bool)
    range_x = np.zeros(len(visible))
    range_y = np.zeros(len(visible))
    range_x[enough] = (rolling(ball_m[:, 0], -np.inf, np.max)[enough]
                       - rolling(ball_m[:, 0], np.inf, np.min)[enough])
    range_y[enough] = (rolling(ball_m[:, 1], -np.inf, np.max)[enough]
                       - rolling(ball_m[:, 1], np.inf, np.min)[enough])
    flags[enough] = (range_x[enough] <= REST_RANGE_M) & (range_y[enough] <= REST_RANGE_M)
    return flags


def stationary_table(atd, other_dead):
    """[{start, stop, dead, reason}] cada período de bola parada, por ordem.

    other_dead: (N,) bool com as paragens já encontradas pelas outras regras, para saber se a
    bola parada está colada a uma delas.
    start/stop já vêm alargados até ao frame em que a bola sai do sítio.
    """
    ball_m = features.ball_metres(atd)
    visible = features.ball_visible(atd)
    flags = stationary_flags(atd)
    nearest, _ = features.nearest_player(atd)
    alone = visible & np.isfinite(nearest) & (nearest > UNATTENDED_M)
    adjacent = _frames(ADJACENT_S)

    out = []
    for half_start, half_stop in match.half_runs():
        starts, stops = features.runs_of(flags[half_start:half_stop])
        for a, b in zip(starts.tolist(), stops.tolist()):
            if b - a < REST_WINDOW_FRAMES:
                continue
            start, stop = half_start + a, half_start + b
            length = stop - start
            n_alone = int(alone[start:stop].sum())

            # Alargar até a bola sair do sítio: é aí que está o pontapé ou o lançamento.
            anchor = np.median(ball_m[start:stop], axis=0)

            def at_rest(frame):
                return bool(visible[frame]) and float(np.hypot(
                    ball_m[frame, 0] - anchor[0], ball_m[frame, 1] - anchor[1])) <= REST_RANGE_M

            while stop < half_stop and at_rest(stop):
                stop += 1
            while start > half_start and at_rest(start - 1):
                start -= 1

            near_other = (other_dead[max(half_start, start - adjacent):start].any()
                          or other_dead[stop:min(half_stop, stop + adjacent)].any()
                          or other_dead[start:stop].any())
            if near_other:
                dead, reason = True, "a) colada a outra paragem"
            elif length >= _frames(MIN_REST_S) and n_alone >= _frames(UNATTENDED_MIN_S):
                dead, reason = True, "b) parada e sozinha"
            elif length >= _frames(LONG_REST_S):
                dead, reason = True, "c) parada muito tempo"
            else:
                dead, reason = False, "d) parada pouco tempo (jogo continua)"
            out.append({"start": start, "stop": stop, "dead": dead, "reason": reason})
    return out


def stationary_mask(atd, other_dead):
    mask = np.zeros(len(atd["ball_xy"]), dtype=bool)
    for item in stationary_table(atd, other_dead):
        if item["dead"]:
            mask[item["start"]:item["stop"]] = True
    return mask


# -- golos --------------------------------------------------------------------------------------

def goal_table(atd, tags, other_dead):
    """[{start, stop, method}] uma paragem por cada "Kick Off (After Goal)" do tagging."""
    out = []
    rows = _set_pieces_of_type(tags, GOAL_KICKOFF_TYPE)
    if rows is None or not len(rows):
        return out

    ball_m = features.ball_metres(atd)
    length, width = float(atd["pitch"][0]), float(atd["pitch"][1])
    in_goal = (np.isfinite(ball_m[:, 0])
               & (np.abs(ball_m[:, 1] - width / 2.0) <= GOAL_HALF_WIDTH_M + GOAL_MARGIN_M)
               & ((ball_m[:, 0] <= GOAL_MARGIN_M) | (ball_m[:, 0] >= length - GOAL_MARGIN_M)))

    for half_start, half_stop in match.half_runs():
        inside = rows[(rows["start_frame"] >= half_start) & (rows["start_frame"] < half_stop)]
        for _, row in inside.iterrows():
            kickoff, _ = kickoff_frame(atd, row["start_frame"], row["end_frame"])
            if kickoff is None:
                continue
            kickoff = min(max(int(kickoff), half_start), half_stop)
            low = max(half_start, kickoff - _frames(GOAL_LOOKBACK_S))

            sightings = low + np.flatnonzero(in_goal[low:kickoff])
            previous = low + np.flatnonzero(other_dead[low:kickoff])
            if len(sightings):
                # Início do último grupo de frames em que a bola esteve na baliza.
                start = int(sightings[-1])
                for frame in sightings[::-1]:
                    if start - int(frame) > _frames(GOAL_CLUSTER_S):
                        break
                    start = int(frame)
                method = "bola vista na baliza"
            elif len(previous):
                start = int(previous[-1])
                method = "desde a última paragem detetada"
            else:
                start = max(low, kickoff - _frames(GOAL_DEFAULT_S))
                method = "{:.0f} s antes do pontapé de saída (sem pistas)".format(GOAL_DEFAULT_S)
            out.append({"start": start, "stop": kickoff, "method": method})
    return out


def goal_mask(atd, tags, other_dead):
    mask = np.zeros(len(atd["ball_xy"]), dtype=bool)
    for goal in goal_table(atd, tags, other_dead):
        mask[goal["start"]:goal["stop"]] = True
    return mask


# -- limpeza ------------------------------------------------------------------------------------

def clean(mask):
    """Tira paragens de menos de MIN_DEAD_S e fecha jogo vivo de menos de MIN_ALIVE_S entre
    duas paragens. Trabalha parte a parte; os extremos de cada parte não são tocados."""
    mask = mask.copy()
    for half_start, half_stop in match.half_runs():
        window = mask[half_start:half_stop]
        n = len(window)

        starts, stops = features.runs_of(window)
        for a, b in zip(starts.tolist(), stops.tolist()):
            if b - a < _frames(MIN_DEAD_S) and a > 0 and b < n:
                window[a:b] = False

        starts, stops = features.runs_of(~window)
        for a, b in zip(starts.tolist(), stops.tolist()):
            if b - a < _frames(MIN_ALIVE_S) and a > 0 and b < n:
                window[a:b] = True
    return mask


# -- tudo junto ---------------------------------------------------------------------------------

def components(atd, tags=None):
    """As peças separadas, para o relatório, e o resultado final."""
    prekickoff = prekickoff_mask(atd, tags)
    outside = outside_mask(atd)
    gaps = gap_mask(atd)
    base = clean(outside | gaps)                     # paragens sem contar com a bola parada
    stationary = stationary_mask(atd, base)
    before_goals = clean(outside | gaps | stationary)
    goals = goal_mask(atd, tags, before_goals)
    final = clean(before_goals | goals | prekickoff) | prekickoff
    return {"prekickoff": prekickoff, "outside": outside, "gaps": gaps, "base": base,
            "stationary": stationary, "goals": goals, "final": final}


def dead_ball_mask(atd, tags=None):
    return components(atd, tags)["final"]