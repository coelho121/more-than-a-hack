"""Momentos do vídeo para a equipa ir VER, para perceber se as regras fazem sentido.

    python scripts/moments.py

Imprime alguns exemplos (no máximo MAX_EXAMPLES por tipo) com o tempo no vídeo tático e,
por baixo de cada um, o que o modelo respondeu nesse bocado (casa, fora ou bola morta), para
se poder comparar com o que se vê.

REGRAS DE UTILIZAÇÃO (docs/oficial/RULES.md):
  - Ver o vídeo para perceber os dados e desenhar regras é permitido.
  - Transformar o que se vê em labels, correções ou constantes ligadas a momentos concretos
    do jogo NÃO é. O que sai daqui só pode ser uma conclusão geral ("fora das áreas, quando a
    bola some, o jogo está quase sempre parado"), nunca "aos 23:41 é bola morta".
  - Por isso mostra só uma amostra e não a lista toda: é para perceber, não para classificar
    os casos um a um.
  - O output deste script fica no terminal. Não se cola num LLM nem se guarda no repo.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from src import data, dead_ball, features, match, pipeline, possession, tagging  # noqa: E402

FPS = 25
MAX_EXAMPLES = 5
LEAD_IN_S = 5      # começar a ver uns segundos antes


NAMES = {"H": "casa", "A": "fora", "D": "bola morta", "X": "sem resposta"}


def short_clock(frame):
    seconds = int(frame // FPS)
    return "{}:{:02d}:{:02d}".format(seconds // 3600, (seconds % 3600) // 60, seconds % 60)


def clock(frame):
    return "{} ({} s)".format(short_clock(frame), int(frame // FPS))


def model_answer(labels, start, stop):
    """O que o modelo respondeu desde LEAD_IN_S antes do momento até LEAD_IN_S depois."""
    low = max(0, start - LEAD_IN_S * FPS)
    high = min(len(labels), stop + LEAD_IN_S * FPS)
    window = labels[low:high]
    breaks = np.flatnonzero(window[1:] != window[:-1]) + 1
    bounds = np.concatenate(([0], breaks, [len(window)]))
    parts = []
    for i in range(len(bounds) - 1):
        name = NAMES.get(str(window[bounds[i]]), str(window[bounds[i]]))
        if i < len(bounds) - 2:
            parts.append("{} até {}".format(name, short_clock(low + int(bounds[i + 1]))))
        else:
            parts.append(name)
    return " -> ".join(parts)


def sample(items):
    """No máximo MAX_EXAMPLES itens, espalhados pelo jogo. Sempre os mesmos (sem sorteio)."""
    if len(items) <= MAX_EXAMPLES:
        return list(items)
    picks = np.linspace(0, len(items) - 1, MAX_EXAMPLES).round().astype(int)
    return [items[i] for i in picks]


def show(title, items, question, labels):
    print("\n== {} ==".format(title))
    print("  O que ver: {}".format(question))
    if not items:
        print("  (nenhum caso)")
    for start, stop in sample(items):
        print("  começa a ver em {}   o momento é {} e dura {:.0f} s".format(
            clock(max(0, start - LEAD_IN_S * FPS)), clock(start), (stop - start) / float(FPS)))
        print("      o modelo diz: {}".format(model_answer(labels, start, stop)))
    if len(items) > MAX_EXAMPLES:
        print("  ({} casos no total, mostrados {})".format(len(items), MAX_EXAMPLES))


def main():
    atd = data.load()
    tags = tagging.load()
    dead = dead_ball.dead_ball_mask(atd, tags)
    labels = pipeline.run(atd, tags)

    # 1. Buracos fora das áreas: a velocidade dos jogadores decide (regras f e g).
    table = dead_ball.gap_table(atd)
    stopped = [(g["start"], g["stop"]) for g in table if g["reason"].startswith("f)")]
    running = [(g["start"], g["stop"]) for g in table if g["reason"].startswith("g)")]
    show("bola some fora das áreas e os jogadores param (regra f, marcámos D)", stopped,
         "o jogo está mesmo parado (falta, fora de jogo, bola na mão)?", labels)
    show("bola some fora das áreas e os jogadores continuam a correr (regra g, jogo vivo)",
         running, "o jogo continua mesmo e a câmara só perdeu a bola?", labels)

    # 2. Troços de jogo vivo sem nenhuma sequência válida.
    codes = possession.control_codes(atd)
    stretches = []
    for half_start, half_stop in match.half_runs():
        starts, stops = features.runs_of(~dead[half_start:half_stop])
        for a, b in zip(starts.tolist(), stops.tolist()):
            start, stop = half_start + a, half_start + b
            if not possession.possession_changes(codes, start, stop):
                stretches.append((start, stop))
    show("jogo vivo em que nenhuma equipa chega a 1 s de bola nos pés", stretches,
         "há mesmo jogo aqui? quem tem a bola? ou é uma paragem que não apanhámos?", labels)
    if stretches:
        length = np.array([b - a for a, b in stretches], dtype=np.float64) / FPS
        print("  AGREGADO (isto pode ir para o Claude): n={}  duração mín {:.1f}s  "
              "mediana {:.1f}s  máx {:.1f}s  | <5s: {}  5-15s: {}  >15s: {}".format(
                  len(length), length.min(), np.median(length), length.max(),
                  int((length < 5).sum()), int(((length >= 5) & (length < 15)).sum()),
                  int((length >= 15).sum())))

    # 3. Pontapés de baliza do Smart Tagging sem nenhuma paragem detetada na janela.
    types = tags["type"].fillna("").str.strip().str.lower()
    goal_kicks = tags[(tags["code"] == "SET PIECES") & (types == "goal kick")]
    missed = [(int(a), int(b)) for a, b in zip(goal_kicks["start_frame"], goal_kicks["end_frame"])
              if not dead[max(0, int(a)):max(0, int(b))].any()]
    show("pontapés de baliza do tagging sem paragem detetada (janela de 35 s)", missed,
         "há mesmo um pontapé de baliza? a bola vê-se parada? o guarda-redes fica em cima dela?",
         labels)
    return 0


if __name__ == "__main__":
    sys.exit(main())
