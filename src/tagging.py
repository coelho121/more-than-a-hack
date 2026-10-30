"""Smart Tagging -> tabela com intervalos em frames do vídeo e a equipa já como H/A.

    from src import tagging
    tags = tagging.load()
    tags[tags.code == "SET PIECES"]

Colunas: code, start_frame, end_frame (o primeiro frame DEPOIS da fase), team ('H', 'A' ou None),
half, type (só para set pieces).

Cuidados (ver docs/oficial/DATA.md):
  - é automático, não é ground truth: tem falsos positivos e falsos negativos;
  - 36 instâncias têm Team = N/A, incluindo TODOS os lançamentos laterais -> team = None;
  - os intervalos são FASES, não posses, e os set pieces começam ~3.5 s antes do reinício.
"""

import pandas as pd

import load_smart_tagging  # do starter/ oficial

# O XML diz o nome do clube; a submissão quer o lado.
HOME_NAMES = ("barcelona",)
AWAY_NAMES = ("atletic club", "athletic club", "athletic")


def _side(team):
    name = (team or "").strip().lower()
    if name in HOME_NAMES:
        return "H"
    if name in AWAY_NAMES:
        return "A"
    return None  # N/A, vazio, ou algo inesperado: tratar explicitamente, nunca assumir


def load(game=None, data_dir=None):
    kwargs = {}
    if game:
        kwargs["game"] = game
    if data_dir:
        kwargs["data_dir"] = data_dir
    events = load_smart_tagging.load(**kwargs)
    return pd.DataFrame({
        "code": [(e["code"] or "").strip().upper() for e in events],
        "start_frame": [e["start_frame"] for e in events],
        "end_frame": [e["end_frame"] for e in events],
        "team": [_side(e["tags"].get("Team")) for e in events],
        "half": [e["tags"].get("Half") for e in events],
        "type": [e["tags"].get("Type") for e in events],
    }).sort_values("start_frame", ignore_index=True)


if __name__ == "__main__":
    tags = load()
    print(tags.code.value_counts().to_string())
    print("\nteam:", tags.team.value_counts(dropna=False).to_dict())
