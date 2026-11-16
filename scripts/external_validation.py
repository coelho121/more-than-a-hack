"""Validação externa da regra de posse, em OUTROS jogos (dados públicos da Metrica Sports).

    git clone https://github.com/metrica-sports/sample-data external/sample-data
    python scripts/external_validation.py

Para que serve: os parâmetros da posse (raio de controlo, tempo de bola nos pés, duração mínima)
não podem ser afinados no jogo do desafio. As regras permitem usar dados de outros jogos, por
isso usamos os Sample Game 1 e 2 que a Metrica publica (tracking a 25 fps + eventos anotados,
jogos anónimos). Fonte: https://github.com/metrica-sports/sample-data

O que faz:
  1. Lê o tracking e os eventos de cada jogo.
  2. Constrói uma posse de referência a partir dos eventos:
       - bola morta = frames em que o tracking (corrigido) não tem bola durante 1 s ou mais;
       - equipa = a do último evento de posse (PASS, SHOT, SET PIECE, RECOVERY, BALL LOST,
         BALL OUT).  Referência "raw": todos os eventos.  Referência "ctrl": uma série de
         eventos seguidos da mesma equipa só conta se tiver um PASS, SHOT ou SET PIECE, ou
         pelo menos 2 eventos (um corte isolado não muda a posse).
  3. Corre a NOSSA regra de posse (src/possession.py) com a bola morta da referência, para
     testar só a posse, e mede o spellF1 oficial contra a referência.
  4. Repete variando um parâmetro de cada vez, com e sem ruído gaussiano nas posições.

Limites: a referência é uma reconstrução nossa a partir de eventos e conta mais posses curtas
do que a definição do desafio; o tracking de exemplo é mais limpo do que o ATD automático.
Por isso os valores finais ficam a meio caminho entre o ótimo daqui e a leitura do CHALLENGE.md.

Nenhum dado do jogo do desafio é usado aqui.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import features, match, possession  # noqa: E402
import possession_metrics as pm  # noqa: E402

DATA = os.path.join(ROOT, "external", "sample-data", "data")
PITCH = np.array([105.0, 68.0])
FPS = 25
NOISE_SEED = 0
POSSESSION_EVENTS = ("PASS", "SHOT", "SET PIECE", "RECOVERY", "BALL LOST", "BALL OUT")
CONTROLLED_EVENTS = ("PASS", "SHOT", "SET PIECE")
PARAMETERS = ("CONTROL_RADIUS_M", "MIN_CONTROL_S", "MIN_SPELL_S")
SWEEPS = (
    ("CONTROL_RADIUS_M", (1.0, 1.5, 1.8, 2.0, 2.5, 3.0)),
    ("MIN_CONTROL_S", (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.3)),
    ("MIN_SPELL_S", (0.0, 2.0, 3.0, 4.0, 5.0)),
)


def read_team(path):
    table = pd.read_csv(path, skiprows=2)
    values = table.iloc[:, 3:].to_numpy(dtype=float)
    return table.iloc[:, 0].to_numpy(), values[:, :-2].reshape(len(table), -1, 2), values[:, -2:]


def load_game(number, noise_m=0.0):
    stem = os.path.join(DATA, "Sample_Game_{0}".format(number), "Sample_Game_{0}_".format(number))
    period, home, ball = read_team(stem + "RawTrackingData_Home_Team.csv")
    _, away, _ = read_team(stem + "RawTrackingData_Away_Team.csv")
    players = np.concatenate([home, away], axis=1)
    team_of_track = np.array([0] * home.shape[1] + [1] * away.shape[1], dtype=np.int8)
    n_first = int((period == 1).sum())
    runs = [(0, n_first), (n_first, len(period))]

    # Bola morta da referência: bola ausente durante 1 s ou mais (buracos menores são falhas).
    dead = ~np.isfinite(ball).all(axis=1)
    for start, stop in runs:
        a, b = features.runs_of(dead[start:stop])
        for x, y in zip(a.tolist(), b.tolist()):
            if y - x < FPS:
                dead[start + x:start + y] = False

    events = pd.read_csv(stem + "RawEventsData.csv")
    events = events[events["Type"].isin(POSSESSION_EVENTS)].sort_values("Start Frame", kind="stable")
    frames = events["Start Frame"].to_numpy() - 1
    teams = (events["Team"] == "Away").to_numpy().astype(np.int8)
    kinds = events["Type"].to_numpy()

    def timeline(event_frames, event_teams):
        index = np.searchsorted(event_frames, np.arange(len(period)), side="right") - 1
        return event_teams[np.clip(index, 0, len(event_teams) - 1)]

    controlled = np.zeros(len(frames), dtype=bool)
    i = 0
    while i < len(frames):
        j = i
        while j + 1 < len(frames) and teams[j + 1] == teams[i]:
            j += 1
        if j > i or kinds[i] in CONTROLLED_EVENTS:
            controlled[i:j + 1] = True
        i = j + 1

    references = {
        "raw": np.where(dead, pm.D, timeline(frames, teams)).astype(np.int8),
        "ctrl": np.where(dead, pm.D, timeline(frames[controlled], teams[controlled])).astype(np.int8),
    }
    if noise_m > 0:
        rng = np.random.RandomState(NOISE_SEED)
        ball = ball + rng.normal(0, noise_m, ball.shape) / PITCH
        players = players + rng.normal(0, noise_m, players.shape) / PITCH
    atd = {"ball_xy": ball, "players_xy": players, "team_of_track": team_of_track, "pitch": PITCH}
    return atd, runs, dead, references


def use_halves(runs):
    """O src/match.py conhece as partes do jogo do desafio; aqui damos-lhe as deste jogo."""
    match.half_runs = lambda game=None: list(runs)


def score(games, reference, **overrides):
    """spellF1 médio dos jogos, com os parâmetros atuais e as alterações pedidas."""
    saved = {name: getattr(possession, name) for name in PARAMETERS}
    for name, value in overrides.items():
        setattr(possession, name, value)
    try:
        values = []
        for atd, runs, dead, references in games:
            use_halves(runs)
            team = possession.assign(atd, dead)
            predicted = np.where(dead, pm.D, team).astype(np.int8)
            values.append(pm.spell_scores(references[reference], predicted, runs)["mean_f1"])
    finally:
        for name, value in saved.items():
            setattr(possession, name, value)
    return float(np.mean(values))


def main():
    if not os.path.isdir(DATA):
        print("Faltam os dados públicos. Corre primeiro:\n"
              "  git clone https://github.com/metrica-sports/sample-data external/sample-data")
        return 1
    original_half_runs = match.half_runs
    try:
        for noise in (0.0, 0.3):
            games = [load_game(1, noise), load_game(2, noise)]
            print("\n===== ruído nas posições: {:.1f} m =====".format(noise))
            for reference in ("raw", "ctrl"):
                current = ", ".join("{}={}".format(n, getattr(possession, n)) for n in PARAMETERS)
                print("-- referência {}:  valores atuais ({})  spellF1 {:.3f}".format(
                    reference, current, score(games, reference)))
                for name, values in SWEEPS:
                    print("   {:<17} {}".format(name, "  ".join(
                        "{}: {:.3f}".format(v, score(games, reference, **{name: v}))
                        for v in values)))
    finally:
        match.half_runs = original_half_runs
    return 0


if __name__ == "__main__":
    sys.exit(main())
