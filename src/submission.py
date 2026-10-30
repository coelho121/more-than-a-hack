"""Escrever o CSV, validá-lo como o scorer oficial e dar score no GT público.

LEMBRETE: o score no GT público (5 minutos, 17 posses) serve para ver se o pipeline está
CERTO, não para escolher entre versões nem para afinar parâmetros. Isso é uma regra
(docs/oficial/RULES.md), não um conselho.
"""

import os

import numpy as np

import possession_metrics as pm  # do evaluation/ oficial
from games import GAME, GAMES
from src import ROOT, RESULTS_DIR

PUBLIC_GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")


def write(labels, name, results_dir=RESULTS_DIR):
    """frame,label CSV com uma linha por frame. Devolve o caminho."""
    expected = GAMES[GAME]["n_video_frames"]
    if len(labels) != expected:
        raise ValueError("{} labels, o vídeo tem {} frames".format(len(labels), expected))
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "{}.csv".format(name))
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, label) for i, label in enumerate(labels))
    return path


def validate(path):
    """None se o ficheiro é válido, senão a razão (as mesmas regras do scorer oficial)."""
    table = pm.read_label_csv(path)
    return pm.validate_submission(table, GAMES[GAME]["n_video_frames"])


def score_public(labels):
    """Os números principais no GT público. Só para verificar que nada está partido."""
    gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
    result = pm.evaluate(gt, pm.label_codes(np.asarray(labels)))
    return {
        "spellF1": round(result["spell_f1_mean"], 4),
        "macroF1": round(result["macro_f1"], 4),
        "posses": result["n_pred_spells"],
        "dur_media_s": round(result["mean_spell_duration_s"], 1),
    }
