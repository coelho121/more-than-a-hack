"""Verificação de definição, feita UMA vez: o que é que "bola morta" quer dizer no GT público?

    python scripts/definition_check.py

Cruza a label do GT público (5 minutos) com "o tracking vê a bola neste frame?" e imprime
uma tabela 3x2 de contagens e percentagens.

Para que serve: responder sim ou não a "um buraco sem bola corresponde a bola morta?".
O CHALLENGE.md diz que o GT público é onde se vê o que a organização entende por "dead".

Para que NÃO serve: nenhum número daqui entra no código, não se comparam versões e não se
volta a correr isto depois de cada alteração. Fica no repo por transparência e está descrito
no documento técnico.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import data, features, submission  # noqa: E402
import possession_metrics as pm  # noqa: E402


def main():
    atd = data.load()
    gt = pm.label_codes(pm.read_label_csv(submission.PUBLIC_GT))
    visible = features.ball_visible(atd)

    rows = []
    for code, name in ((pm.D, "D"), (pm.H, "H"), (pm.A, "A")):
        is_label = gt == code
        rows.append((name, int((is_label & visible).sum()), int((is_label & ~visible).sum())))

    total_vis = sum(r[1] for r in rows)
    total_inv = sum(r[2] for r in rows)
    print("GT público: {} frames com label".format(total_vis + total_inv))

    print("\nPor linha (cada linha soma 100%): quando a resposta certa é X, a bola via-se?")
    print("  {:<6} {:>10} {:>14}".format("label", "visível", "não visível"))
    for name, vis, inv in rows:
        n = vis + inv
        print("  {:<6} {:>9.1f}% {:>13.1f}%   (n={})".format(
            name, 100.0 * vis / n if n else 0.0, 100.0 * inv / n if n else 0.0, n))

    print("\nPor coluna (cada coluna soma 100%): quando a bola (não) se vê, qual é a resposta?")
    print("  {:<6} {:>10} {:>14}".format("label", "visível", "não visível"))
    for name, vis, inv in rows:
        print("  {:<6} {:>9.1f}% {:>13.1f}%".format(
            name, 100.0 * vis / total_vis if total_vis else 0.0,
            100.0 * inv / total_inv if total_inv else 0.0))
    print("  {:<6} {:>10} {:>14}".format("n", total_vis, total_inv))
    return 0


if __name__ == "__main__":
    sys.exit(main())