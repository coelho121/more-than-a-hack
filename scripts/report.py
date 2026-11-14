"""Relatório de plausibilidade: só números agregados, nunca dados do jogo frame a frame.

    python scripts/report.py

Imprime, para a baseline e para o pipeline atual: distribuição de labels, número e duração das
posses e das paragens. Depois, alguns agregados do tracking e do Smart Tagging que servem para
escolher parâmetros com lógica de futebol (e não pelo GT público).

O score no GT público aparece no fim SÓ como alarme de "partiu-se alguma coisa".
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from src import data, dead_ball, features, match, pipeline, possession  # noqa: E402
from src import submission, tagging  # noqa: E402
import possession_metrics as pm  # noqa: E402
import load_smart_tagging as official_tagging  # noqa: E402

FPS = 25


def runs_of(mask):
    """(inícios, fins) dos troços consecutivos de True."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def durations_line(name, frames):
    if not len(frames):
        print("  {:<10} n=   0".format(name))
        return
    s = np.asarray(frames, dtype=np.float64) / FPS
    print("  {:<10} n={:>4}  média {:>5.1f}s  mediana {:>5.1f}s  p10 {:>4.1f}s  p90 {:>5.1f}s  "
          "máx {:>6.1f}s  | <1s: {}  <2s: {}  <3s: {}  >60s: {}".format(
              name, len(s), s.mean(), np.median(s), np.percentile(s, 10), np.percentile(s, 90),
              s.max(), int((s < 1).sum()), int((s < 2).sum()), int((s < 3).sum()),
              int((s > 60).sum())))


def label_summary(name, labels):
    labels = np.asarray(labels)
    runs = match.half_runs()
    graded = match.graded_mask(len(labels))
    raw = pm.label_codes(labels)
    codes = pm.fill_abstentions(raw, runs)

    print("\n== {} ==".format(name))
    n = int(graded.sum())
    counts = {k: int(np.sum(codes[graded] == v)) for k, v in
              (("H", pm.H), ("A", pm.A), ("D", pm.D), ("X", pm.X))}
    print("  frames avaliados: {}   ".format(n) + "   ".join(
        "{} {:.1f}%".format(k, 100.0 * v / n) for k, v in counts.items()))
    alive = counts["H"] + counts["A"]
    print("  quota da casa no tempo vivo: {:.1f}%   tempo vivo: {:.1f} min".format(
        100.0 * counts["H"] / alive if alive else 0.0, alive / FPS / 60.0))
    outside = ~graded
    print("  fora das partes: {} frames, {} com X".format(
        int(outside.sum()), int(np.sum(raw[outside] == pm.X))))

    segs = pm.segments(codes, runs)
    durations_line("posses", [b - a for a, b, k in segs if k in (pm.H, pm.A)])
    durations_line("paragens", [b - a for a, b, k in segs if k == pm.D])
    print("  GT público (só alarme):", submission.score_public(labels))


def kickoff_summary(atd, tags):
    print("\n== pontapés de saída detetados ==")
    published = official_tagging.KICKOFF_VIDEO_FRAMES  # publicado no starter oficial
    names = {1: "1st Half", 2: "2nd Half"}
    found = dead_ball.kickoffs(atd, tags)
    if not found:
        print("  nenhum (o tagging não tem Kick Off (Start)?)")
    for k in found:
        ref = published.get(names.get(k["half"]))
        diff = "" if ref is None else "   publicado {}   diferença {:+d} frames".format(
            ref, k["frame"] - ref)
        print("  parte {}: frame {}{}   [{}]   D acrescentado: {} frames".format(
            k["half"], k["frame"], diff, k["method"], k["frame"] - k["first"]))


def tracking_summary(atd):
    print("\n== tracking (só frames avaliados) ==")
    graded = match.graded_mask(len(atd["ball_xy"]))
    n = int(graded.sum())
    visible = features.ball_visible(atd)
    gv = graded & visible
    print("  bola visível: {:.1f}%".format(100.0 * gv.sum() / n))

    lengths = []
    for start, stop in match.half_runs():
        a, b = runs_of(~visible[start:stop])
        lengths.extend((b - a).tolist())
    lengths = np.asarray(lengths, dtype=np.float64) / FPS
    print("  buracos sem bola: {}".format(len(lengths)))
    for low, high in ((0, 0.5), (0.5, 1), (1, 2), (2, 5), (5, 10), (10, 30), (30, 1e9)):
        pick = (lengths >= low) & (lengths < high)
        label = "{}-{}s".format(low, high) if high < 1e9 else ">{}s".format(low)
        print("    {:<8} n={:>5}   {:>5.1f}% dos frames avaliados".format(
            label, int(pick.sum()), 100.0 * lengths[pick].sum() * FPS / n))

    dist, _ = features.nearest_player(atd)
    d = dist[gv & np.isfinite(dist)]
    if len(d):
        print("  distância do jogador mais perto à bola (m): " + "  ".join(
            "p{} {:.2f}".format(p, np.percentile(d, p)) for p in (25, 50, 75, 90, 95)))
        print("    " + "  ".join("<{}m {:.1f}%".format(t, 100.0 * (d < t).mean())
                                 for t in (1, 1.5, 2, 3, 5)))

    codes = possession.nearest_player_codes(atd)
    c = codes[graded & (codes >= 0)]
    flips = int(np.sum(c[1:] != c[:-1]))
    print("  trocas da equipa mais perto (bruto): {}   ({:.1f} por minuto com bola visível)".format(
        flips, flips / (len(c) / FPS / 60.0) if len(c) else 0.0))

    xy = atd["ball_xy"][gv]
    out = ((xy < 0) | (xy > 1)).any(axis=1)
    print("  bola fora de [0,1]: {:.2f}% dos frames com bola".format(100.0 * out.mean()))
    for margin in (0.01, 0.02):
        near = ((xy < margin) | (xy > 1 - margin)).any(axis=1)
        print("  bola a menos de {:.0f}% de uma linha (ou fora): {:.2f}%".format(
            100 * margin, 100.0 * near.mean()))

    speed = atd["ball_s"][gv]
    speed = speed[np.isfinite(speed)]
    if len(speed):
        print("  velocidade da bola (unidades do ficheiro): " + "  ".join(
            "p{} {:.2f}".format(p, np.percentile(speed, p)) for p in (10, 25, 50, 75, 90, 99)))
        print("    " + "  ".join("<{} {:.1f}%".format(t, 100.0 * (speed < t).mean())
                                 for t in (0.5, 1, 2)))
    else:
        print("  velocidade da bola: sem valores")

    tracked = np.isfinite(atd["players_xy"][..., 0])[graded].sum(axis=1)
    print("  jogadores seguidos por frame: média {:.1f}  p5 {:.0f}  p95 {:.0f}".format(
        tracked.mean(), np.percentile(tracked, 5), np.percentile(tracked, 95)))
    ps = atd["players_s"][graded]
    ps = ps[np.isfinite(ps)]
    if len(ps):
        print("  velocidade dos jogadores (m/s): " + "  ".join(
            "p{} {:.2f}".format(p, np.percentile(ps, p)) for p in (25, 50, 75, 90, 99)))


def tagging_summary(tags):
    print("\n== Smart Tagging: set pieces ==")
    sp = tags[tags["code"] == "SET PIECES"]
    length = (sp["end_frame"] - sp["start_frame"]) / FPS
    for kind in sorted(sp["type"].fillna("?").unique()):
        pick = sp["type"].fillna("?") == kind
        print("  {:<24} n={:>3}  duração mediana {:>4.1f}s  (mín {:.1f}s, máx {:.1f}s)  "
              "sem equipa: {}".format(kind, int(pick.sum()), length[pick].median(),
                                      length[pick].min(), length[pick].max(),
                                      int(sp[pick]["team"].isna().sum())))
    graded = match.graded_mask()
    covered = np.zeros(len(graded), dtype=bool)
    for a, b in zip(tags["start_frame"], tags["end_frame"]):
        covered[max(0, int(a)):max(0, int(b))] = True
    print("  frames avaliados cobertos por alguma fase do tagging: {:.1f}%".format(
        100.0 * (covered & graded).sum() / graded.sum()))


def main():
    atd = data.load()
    tags = tagging.load()

    label_summary("baseline_smooth50 (referência)", pipeline.baseline(atd, 50))
    label_summary("pipeline atual", pipeline.run(atd, tags))
    kickoff_summary(atd, tags)
    tracking_summary(atd)
    tagging_summary(tags)
    return 0


if __name__ == "__main__":
    sys.exit(main())