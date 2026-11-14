"""Relatório de plausibilidade: só números agregados, nunca dados do jogo frame a frame.

    python scripts/report.py

Imprime, para a baseline e para o pipeline atual: distribuição de labels, número e duração das
posses e das paragens. Depois mostra de onde vem a bola morta, como foram classificados os
buracos sem bola, se os set pieces do Smart Tagging coincidem com paragens detetadas, e alguns
números internos da posse.

Serve para escolher parâmetros com lógica de futebol (e não pelo GT público). O score no GT
público aparece SÓ como alarme de "partiu-se alguma coisa".
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
    # Posses por parte, para ver se as duas partes são parecidas.
    for index, (start, stop) in enumerate(runs, 1):
        n_half = sum(1 for a, b, k in segs if k in (pm.H, pm.A) and start <= a < stop)
        print("  parte {}: {} posses".format(index, n_half))
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
        print("  parte {}: frame {}{}   [{}]".format(k["half"], k["frame"], diff, k["method"]))


def dead_summary(atd, tags):
    print("\n== de onde vem a bola morta (% dos frames avaliados) ==")
    graded = match.graded_mask(len(atd["ball_xy"]))
    n = float(graded.sum())
    parts = dead_ball.components(atd, tags)
    visible = features.ball_visible(atd)
    for key, name in (("prekickoff", "antes do pontapé de saída"),
                      ("outside", "bola vista fora do campo"),
                      ("gaps", "buracos classificados como paragem"),
                      ("final", "TOTAL depois da limpeza")):
        print("  {:<36} {:>5.1f}%".format(name, 100.0 * (parts[key] & graded).sum() / n))
    final = parts["final"] & graded
    print("  do D total, com bola visível: {:.1f}%".format(
        100.0 * (final & visible).sum() / max(1, final.sum())))
    return parts["final"]


def gap_summary(atd, tags):
    print("\n== buracos sem bola (>= {:.0f} s), por classificação ==".format(dead_ball.MIN_GAP_S))
    gaps = dead_ball.gap_table(atd)
    set_pieces = tags[tags["code"] == "SET PIECES"]
    windows = list(zip(set_pieces["start_frame"], set_pieces["end_frame"]))

    reasons = []
    for gap in gaps:
        if gap["reason"] not in reasons:
            reasons.append(gap["reason"])
    for reason in sorted(reasons):
        chosen = [g for g in gaps if g["reason"] == reason]
        length = np.array([g["stop"] - g["start"] for g in chosen], dtype=np.float64) / FPS
        in_window = sum(1 for g in chosen
                        if any(g["start"] < b and a < g["stop"] for a, b in windows))
        print("  {:<42} {:<4}  n={:>3}  total {:>6.0f}s  mediana {:>5.1f}s  "
              "5-10s: {}  10-30s: {}  >30s: {}  | tocam num set piece: {}".format(
                  reason, "D" if chosen[0]["dead"] else "vivo", len(chosen), length.sum(),
                  np.median(length), int(((length >= 5) & (length < 10)).sum()),
                  int(((length >= 10) & (length < 30)).sum()), int((length >= 30).sum()),
                  in_window))


def set_piece_summary(tags, dead):
    print("\n== set pieces do Smart Tagging vs paragens detetadas ==")
    sp = tags[tags["code"] == "SET PIECES"]
    kinds = sp["type"].fillna("?")
    for kind in sorted(kinds.unique()):
        rows = sp[kinds == kind]
        with_dead, share = 0, []
        for a, b in zip(rows["start_frame"], rows["end_frame"]):
            window = dead[max(0, int(a)):max(0, int(b))]
            if len(window):
                with_dead += int(window.any())
                share.append(window.mean())
        print("  {:<24} n={:>3}  com paragem na janela: {:>3}  "
              "parte da janela em D (média): {:>4.0f}%".format(
                  kind, len(rows), with_dead, 100.0 * np.mean(share) if share else 0.0))


def possession_summary(atd, dead):
    print("\n== posse: números internos (só jogo vivo, frames avaliados) ==")
    graded = match.graded_mask(len(dead))
    alive = graded & ~dead
    visible = features.ball_visible(atd)
    codes = possession.control_codes(atd)
    n_visible = float((alive & visible).sum())
    print("  jogo vivo com bola visível: {:.1f}% do jogo vivo".format(
        100.0 * n_visible / max(1, alive.sum())))
    print("  nesses frames: controlo casa {:.1f}%  fora {:.1f}%  ninguém {:.1f}%".format(
        100.0 * (alive & visible & (codes == 0)).sum() / max(1.0, n_visible),
        100.0 * (alive & visible & (codes == 1)).sum() / max(1.0, n_visible),
        100.0 * (alive & visible & (codes == -1)).sum() / max(1.0, n_visible)))

    n_touches = n_sequences = n_valid = n_stretches = n_fallback = 0
    min_frames = int(round(possession.MIN_CONTROL_S * FPS))
    for half_start, half_stop in match.half_runs():
        starts, stops = features.runs_of(~dead[half_start:half_stop])
        for a, b in zip(starts.tolist(), stops.tolist()):
            start, stop = half_start + a, half_start + b
            n_stretches += 1
            n_touches += len(possession.touches(codes, start, stop))
            seqs = possession.sequences(codes, start, stop)
            n_sequences += len(seqs)
            n_valid += sum(1 for _, _, control in seqs if control >= min_frames)
            if not possession.possession_changes(codes, start, stop):
                n_fallback += 1
    print("  troços de jogo vivo: {}   (sem nenhuma sequência válida: {})".format(
        n_stretches, n_fallback))
    print("  toques: {}   sequências: {}   sequências válidas (>= {:.1f} s): {}".format(
        n_touches, n_sequences, possession.MIN_CONTROL_S, n_valid))


def main():
    atd = data.load()
    tags = tagging.load()

    label_summary("baseline_smooth50 (referência)", pipeline.baseline(atd, 50))
    label_summary("pipeline atual", pipeline.run(atd, tags))
    kickoff_summary(atd, tags)
    dead = dead_summary(atd, tags)
    gap_summary(atd, tags)
    set_piece_summary(tags, dead)
    possession_summary(atd, dead)
    return 0


if __name__ == "__main__":
    sys.exit(main())