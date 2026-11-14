"""Diagnóstico: a velocidade dos jogadores distingue "jogo parado" de "jogo a decorrer"?

    python scripts/speed_diagnostic.py

A ideia a testar: numa paragem os jogadores andam a passo; numa jogada correm. Se isso se vir
nos dados, a velocidade pode decidir os buracos sem bola em que a posição no campo não chega.

"Velocidade coletiva" de um frame = mediana da velocidade (m/s) dos jogadores seguidos nesse
frame. A mediana ignora os casos extremos (um guarda-redes parado, um extremo em sprint).

Compara três grupos, sempre com a média da velocidade coletiva num intervalo de vários segundos:
  CERTO PARADO   buracos em que a bola saiu de certeza (regras b, c e d: junto a uma linha)
  CERTO JOGO     janelas de 6 s de jogo vivo com a bola quase sempre à vista
  DUVIDOSOS      buracos das regras e (mais de 15 s), f e g (fora das áreas) e h (numa área)

Se CERTO PARADO e CERTO JOGO ficarem em sítios diferentes da tabela, a velocidade serve, e a
linha dos DUVIDOSOS mostra para que lado cai cada um. Se ficarem misturados, não serve.

Resultado neste jogo: mediana 1.15 m/s nos buracos certos e 2.0 m/s no jogo corrido. Com o
limite em 1.5 m/s (passo de caminhada) erra-se cerca de 20% para cada lado. É daqui que vem o
WALKING_SPEED_MS de src/dead_ball.py, usado só nos buracos fora das áreas (regras f e g).

Não usa ground truth nem nada visto no vídeo. Só números agregados.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from src import data, dead_ball, features, match, tagging  # noqa: E402

FPS = 25
WINDOW_S = 6
EDGES = (0.0, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 99.0)


def header():
    labels = ["<{:.1f}".format(EDGES[1])]
    labels += ["{:.1f}-{:.1f}".format(a, b) for a, b in zip(EDGES[1:-2], EDGES[2:-1])]
    labels += [">{:.1f}".format(EDGES[-2])]
    print("  {:<34} {:>4}  {}   | {}".format(
        "grupo", "n", "  ".join("{:>7}".format(x) for x in labels), "p10   p50   p90"))


def row(name, values):
    values = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if not len(values):
        print("  {:<34} {:>4}".format(name, 0))
        return
    counts, _ = np.histogram(values, bins=EDGES)
    print("  {:<34} {:>4}  {}   | {:.2f}  {:.2f}  {:.2f}".format(
        name, len(values), "  ".join("{:>7}".format(int(c)) for c in counts),
        np.percentile(values, 10), np.percentile(values, 50), np.percentile(values, 90)))


def profile(name, speed, anchors, half_bounds):
    """Velocidade coletiva mediana em fatias de 2 s, de -10 s a +10 s à volta de cada âncora."""
    cells = []
    for low in range(-10, 10, 2):
        values = []
        for frame in anchors:
            start, stop = frame + low * FPS, frame + (low + 2) * FPS
            if any(a <= start and stop <= b for a, b in half_bounds):
                values.append(features.interval_mean(speed, start, stop))
        values = [v for v in values if np.isfinite(v)]
        cells.append("{:>5.2f}".format(np.median(values)) if values else "    -")
    print("  {:<34} {}".format(name, " ".join(cells)))


def main():
    atd = data.load()
    tags = tagging.load()
    speed = features.collective_speed(atd)
    visible = features.ball_visible(atd)
    dead = dead_ball.dead_ball_mask(atd, tags)
    graded = match.graded_mask(len(speed))
    halves = match.half_runs()

    print("== velocidade coletiva (m/s), todos os frames avaliados ==")
    all_speed = speed[graded & np.isfinite(speed)]
    print("  " + "  ".join("p{} {:.2f}".format(p, np.percentile(all_speed, p))
                           for p in (5, 25, 50, 75, 95)))

    gaps = dead_ball.gap_table(atd)
    by_rule = {}
    for gap in gaps:
        by_rule.setdefault(gap["reason"][:2], []).append(gap)

    def gap_means(rules):
        return [features.interval_mean(speed, g["start"], g["stop"])
                for rule in rules for g in by_rule.get(rule, [])]

    # Janelas de jogo vivo: 6 s seguidos sem bola morta e com a bola à vista em 80% dos frames.
    alive_windows = []
    size = WINDOW_S * FPS
    for half_start, half_stop in halves:
        starts, stops = features.runs_of(~dead[half_start:half_stop])
        for a, b in zip(starts.tolist(), stops.tolist()):
            for start in range(half_start + a, half_start + b - size + 1, size):
                if visible[start:start + size].mean() >= 0.8:
                    alive_windows.append(features.interval_mean(speed, start, start + size))

    print("\n== média da velocidade coletiva por intervalo: quantos caem em cada faixa (m/s) ==")
    header()
    row("CERTO PARADO  (buracos b, c, d)", gap_means(("b)", "c)", "d)")))
    row("CERTO JOGO    (janelas de 6 s)", alive_windows)
    print()
    row("duvidosos e)  mais de 15 s", gap_means(("e)",)))
    row("duvidosos f+g) fora das áreas", gap_means(("f)", "g)")))
    row("duvidosos h)  numa área", gap_means(("h)",)))

    print("\n== o mesmo, só para os buracos CERTO PARADO, por duração ==")
    header()
    sure = [g for rule in ("b)", "c)", "d)") for g in by_rule.get(rule, [])]
    for name, low, high in (("5 a 10 s", 5, 10), ("10 a 20 s", 10, 20), ("mais de 20 s", 20, 1e9)):
        chosen = [features.interval_mean(speed, g["start"], g["stop"]) for g in sure
                  if low <= (g["stop"] - g["start"]) / float(FPS) < high]
        row("CERTO PARADO  " + name, chosen)

    print("\n== como a velocidade evolui à volta de um buraco CERTO PARADO ==")
    print("  fatias de 2 s:                     " + " ".join(
        "{:>5}".format("{:+d}".format(low)) for low in range(-10, 10, 2)))
    profile("à volta do INÍCIO do buraco", speed, [g["start"] for g in sure], halves)
    profile("à volta do FIM do buraco", speed, [g["stop"] for g in sure], halves)
    print("  (-10 = dez segundos antes; +8 = fatia que começa oito segundos depois)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
