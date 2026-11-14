"""Teste de robustez: o resultado aguenta-se se mexermos um bocado em cada parâmetro?

    python scripts/sensitivity.py

Para cada parâmetro do método, corre o pipeline com o valor 30% abaixo e 30% acima (um
parâmetro de cada vez, os outros ficam no valor normal) e compara com a versão normal.

Não usa ground truth nenhum. A comparação é sempre contra a NOSSA versão normal:
  - posses, tempo vivo e % de D: mudam muito ou pouco?
  - "iguais": % dos frames avaliados que ficam com a mesma label;
  - "spellF1 vs normal": o spellF1 oficial, mas calculado contra a nossa versão normal em vez
    do ground truth. 1.00 = as posses são as mesmas; quanto mais baixo, mais posses mudaram.

Para que serve: um parâmetro em que +-30% quase não muda nada é uma escolha segura. Um
parâmetro em que +-30% muda muitas posses é uma escolha frágil, e convém ter uma boa razão de
futebol para o valor que lá está. Só números agregados.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from src import data, dead_ball, match, pipeline, possession, tagging  # noqa: E402
import possession_metrics as pm  # noqa: E402

FPS = 25
CHANGE = 0.30

# (módulo, nome do parâmetro, é um número inteiro de frames?)
PARAMETERS = (
    (possession, "CONTROL_RADIUS_M", False),
    (possession, "CONTROL_MARGIN_M", False),
    (possession, "MIN_TOUCH_FRAMES", True),
    (possession, "MIN_CONTROL_S", False),
    (possession, "MIN_SPELL_S", False),
    (dead_ball, "OUT_MARGIN_M", False),
    (dead_ball, "MIN_GAP_S", False),
    (dead_ball, "EDGE_M", False),
    (dead_ball, "LOOKAHEAD_S", False),
    (dead_ball, "MAX_HIDDEN_S", False),
    (dead_ball, "REST_RANGE_M", False),
    (dead_ball, "UNATTENDED_M", False),
    (dead_ball, "MIN_REST_S", False),
    (dead_ball, "LONG_REST_S", False),
    (dead_ball, "ADJACENT_S", False),
    (dead_ball, "MIN_DEAD_S", False),
    (dead_ball, "MIN_ALIVE_S", False),
    (dead_ball, "CENTRE_RADIUS_M", False),
)


def summary(labels, reference_codes=None):
    codes = pm.label_codes(labels)
    graded = match.graded_mask(len(codes))
    n = float(graded.sum())
    alive = (codes == pm.H) | (codes == pm.A)
    spells = [1 for a, b, k in pm.segments(codes, match.half_runs()) if k in (pm.H, pm.A)]
    out = {
        "codes": codes,
        "posses": len(spells),
        "vivo_min": (alive & graded).sum() / FPS / 60.0,
        "dead_pct": 100.0 * ((codes == pm.D) & graded).sum() / n,
    }
    if reference_codes is not None:
        out["iguais"] = 100.0 * ((codes == reference_codes) & graded).sum() / n
        out["spell"] = pm.evaluate(reference_codes, codes)["spell_f1_mean"]
    return out


def line(name, value, s):
    extra = ""
    if "iguais" in s:
        extra = "   iguais {:>5.1f}%   spellF1 vs normal {:.2f}".format(s["iguais"], s["spell"])
    print("  {:<18} {:>6}   posses {:>4}   vivo {:>5.1f} min   D {:>4.1f}%{}".format(
        name, value, s["posses"], s["vivo_min"], s["dead_pct"], extra))


def main():
    atd = data.load()
    tags = tagging.load()

    normal = summary(pipeline.run(atd, tags))
    print("== versão normal ==")
    line("(todos)", "", normal)

    print("\n== um parâmetro de cada vez, -30% e +30% ==")
    worst = []
    for module, name, is_int in PARAMETERS:
        original = getattr(module, name)
        for factor in (1.0 - CHANGE, 1.0 + CHANGE):
            value = original * factor
            if is_int:
                value = max(1, int(round(value)))
                if value == original:
                    value = original + (1 if factor > 1 else -1)
            setattr(module, name, value)
            try:
                s = summary(pipeline.run(atd, tags), normal["codes"])
            finally:
                setattr(module, name, original)
            shown = str(value) if is_int else "{:.2f}".format(value)
            line(name, shown, s)
            worst.append((s["spell"], name, shown))
        print()

    print("== os 5 casos que mais mudam as posses ==")
    for spell, name, shown in sorted(worst)[:5]:
        print("  {:<18} {:>6}   spellF1 vs normal {:.2f}".format(name, shown, spell))

    # Confirmar que no fim ficou tudo como estava.
    again = summary(pipeline.run(atd, tags), normal["codes"])
    print("\nparâmetros repostos: {}".format(
        "OK" if np.array_equal(again["codes"], normal["codes"]) else "ERRO"))
    return 0


if __name__ == "__main__":
    sys.exit(main())