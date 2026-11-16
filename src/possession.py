"""Frente 2: quem tem a bola (H/A) e quando é que a posse muda.

Estado atual (passo 2): posse com memória.

A regra do desafio (docs/oficial/CHALLENGE.md): a posse só muda quando o adversário ganha a
bola de forma CONTROLADA. Um toque, um desvio ou um duelo aéreo não chegam, e numa bola
dividida a posse fica com quem a tinha.

Como isso está implementado:

  1. CONTROLO NUM FRAME. Uma equipa controla a bola num frame se o seu jogador mais próximo
     está a menos de CONTROL_RADIUS_M da bola e claramente mais perto do que o adversário
     mais próximo (CONTROL_MARGIN_M). Se os dois estão em cima da bola, ninguém controla.

  2. TOQUES. Frames seguidos de controlo da mesma equipa formam um toque. Toques de menos de
     MIN_TOUCH_FRAMES são ruído e ignoram-se.

  3. SEQUÊNCIAS. Toques seguidos da mesma equipa, sem nenhum toque do adversário pelo meio,
     formam uma sequência. Uma sequência só conta como posse controlada se somar pelo menos
     MIN_CONTROL_S de controlo. As outras (um corte, um desvio, a bola a passar perto de um
     adversário) são apagadas e a posse não muda.

  4. TROCA. A posse muda no INÍCIO da sequência válida da outra equipa, e não quando a
     confirmamos: o jogo é processado todo de uma vez, por isso podemos olhar para a frente.
     Isto importa porque o spellF1 castiga muito os atrasos.

  5. DURAÇÃO MÍNIMA. Uma posse que dure menos de MIN_SPELL_S entre duas posses do
     adversário é apagada (bola recuperada e logo perdida). A primeira e a última posse de
     cada troço de jogo vivo ficam sempre.

  6. REINÍCIO. Depois de cada paragem a posse começa de novo: do reinício até à primeira
     sequência válida, a bola é da equipa dessa sequência.

Enquanto a bola não se vê mas o jogo continua, não há toques, por isso a posse mantém-se.

As funções nearest_player_codes e majority_smooth são a baseline oficial e ficam cá só para o
teste de sanidade (scripts/check_reproduces_baseline.py).
"""

import numpy as np

from src import features, match

FPS = features.FPS
NOBODY = -1

# Bola "nos pés": o tracking dá o centro do jogador, e uma bola conduzida anda até 1.5-2 m à
# frente dele. 1.8 m apanha quase toda a condução e reduz os casos em que um adversário que só
# está por perto conta como tendo a bola. Escolhido por lógica de futebol, sem olhar para scores.
CONTROL_RADIUS_M = 1.8
# Se o adversário mais próximo está praticamente à mesma distância, é uma bola dividida.
CONTROL_MARGIN_M = 0.3
# Menos de 3 frames (0.12 s) seguidos é ruído do tracking, não um toque.
MIN_TOUCH_FRAMES = 3
# "Controlada" = pelo menos 0.6 s de bola nos pés, somado ao longo da sequência. Uma bola a
# 10 m/s que passa a 1 m de um adversário fica uns 0.35 s dentro do raio dele; um corte ou um
# desvio fica menos. 0.6 s exige dois ou três toques a sério, ou um jogador a segurar a bola.
MIN_CONTROL_S = 0.6
# Uma posse no meio do jogo tem de durar pelo menos 2 s de relógio, senão não é posse: é uma
# bola recuperada e logo perdida. O CHALLENGE.md fala em "manter a bola alguns segundos".
# Atenção à diferença: MIN_CONTROL_S conta só o tempo com a bola nos pés; isto conta o tempo
# todo desde que a equipa ganha a bola até a perder, incluindo os passes em viagem.
MIN_SPELL_S = 2.0


# -- baseline oficial (só para o teste de sanidade) ---------------------------------------------

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


# -- posse com memória --------------------------------------------------------------------------

def control_codes(atd):
    """(N,) equipa que controla a bola em cada frame: 0 casa, 1 fora, -1 ninguém."""
    home, away = features.team_nearest_distances(atd)
    codes = np.full(len(home), NOBODY, dtype=np.int8)
    codes[(home <= CONTROL_RADIUS_M) & (home + CONTROL_MARGIN_M <= away)] = 0
    codes[(away <= CONTROL_RADIUS_M) & (away + CONTROL_MARGIN_M <= home)] = 1
    return codes


def touches(codes, start, stop):
    """[(início, fim, equipa)] toques dentro de [start, stop), por ordem."""
    out = []
    window = codes[start:stop]
    if not len(window):
        return out
    breaks = np.flatnonzero(window[1:] != window[:-1]) + 1
    bounds = np.concatenate(([0], breaks, [len(window)]))
    for i in range(len(bounds) - 1):
        a, b = int(bounds[i]), int(bounds[i + 1])
        team = int(window[a])
        if team != NOBODY and b - a >= MIN_TOUCH_FRAMES:
            out.append((start + a, start + b, team))
    return out


def sequences(codes, start, stop):
    """[(início, equipa, frames_de_controlo)] toques seguidos da mesma equipa, agrupados."""
    out = []
    for a, b, team in touches(codes, start, stop):
        if out and out[-1][1] == team:
            out[-1][2] += b - a
        else:
            out.append([a, team, b - a])
    return [tuple(item) for item in out]


def possession_changes(codes, start, stop):
    """[(frame, equipa)] onde a posse passa a ser de cada equipa dentro de [start, stop).

    Vazio se nenhuma sequência chegar a MIN_CONTROL_S.
    """
    min_frames = int(round(MIN_CONTROL_S * FPS))
    changes = []
    for frame, team, control in sequences(codes, start, stop):
        if control < min_frames:
            continue                      # toque solto: a posse não muda
        if changes and changes[-1][1] == team:
            continue                      # a mesma equipa continua com a bola
        changes.append((frame, team))
    return _drop_short_spells(changes, start, stop)


def _drop_short_spells(changes, start, stop):
    """Apaga posses de menos de MIN_SPELL_S que estão ENTRE duas posses do adversário.

    A primeira e a última posse de cada troço de jogo vivo não são tocadas: depois de um
    reinício há sempre um possuidor claro, mesmo que perca a bola logo a seguir, e uma posse
    que acaba numa paragem acabou mesmo.

    Apaga-se uma de cada vez, a mais curta primeiro (em empate, a que vem primeiro), porque
    ao apagar uma posse as duas vizinhas juntam-se numa só.
    """
    min_frames = int(round(MIN_SPELL_S * FPS))
    changes = list(changes)
    while len(changes) >= 3:
        frames = [start] + [frame for frame, _ in changes[1:]] + [stop]
        lengths = [frames[i + 1] - frames[i] for i in range(len(changes))]
        interior = [(lengths[i], i) for i in range(1, len(changes) - 1)
                    if lengths[i] < min_frames]
        if not interior:
            break
        _, index = min(interior)
        del changes[index:index + 2]      # a posse curta e o regresso do adversário
    return changes


def _fallback_team(codes, nearest, start, stop, previous):
    """Equipa para um troço sem nenhuma sequência válida. Ordem de preferência: quem controlou
    mais frames, quem esteve mais vezes mais perto da bola, quem tinha a bola antes, casa."""
    for source in (codes, nearest):
        window = source[start:stop]
        home, away = int(np.sum(window == 0)), int(np.sum(window == 1))
        if home != away:
            return 0 if home > away else 1
    return previous if previous in (0, 1) else 0


def assign(atd, dead):
    """(N,) int8: 0 casa, 1 fora nos frames de jogo vivo; -1 nos de bola morta e fora das partes.

    dead: (N,) bool de src/dead_ball.py. Cada troço de jogo vivo entre duas paragens é tratado
    por si, porque depois de uma paragem a posse recomeça.
    """
    codes = control_codes(atd)
    nearest = nearest_player_codes(atd)
    team = np.full(len(codes), NOBODY, dtype=np.int8)

    for half_start, half_stop in match.half_runs():
        previous = NOBODY
        starts, stops = features.runs_of(~dead[half_start:half_stop])
        for a, b in zip(starts.tolist(), stops.tolist()):
            start, stop = half_start + a, half_start + b
            changes = possession_changes(codes, start, stop)
            if not changes:
                previous = _fallback_team(codes, nearest, start, stop, previous)
                team[start:stop] = previous
                continue
            # Do reinício até à primeira sequência válida a bola é dessa equipa.
            frames = [start] + [frame for frame, _ in changes[1:]] + [stop]
            for i, (_, side) in enumerate(changes):
                team[frames[i]:frames[i + 1]] = side
            previous = changes[-1][1]
    return team