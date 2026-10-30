"""Teste de sanidade: o nosso pipeline v0 tem de dar EXATAMENTE a baseline_smooth50 oficial.

    python scripts/check_reproduces_baseline.py

Se der igual: a leitura dos dados, o alinhamento dos frames e a escrita do CSV estão certos.
Se não der: há um bug na base, e tudo o que se construir por cima está torto. Corre isto
sempre que mexeres em src/data.py, src/features.py ou src/submission.py.
"""

import filecmp
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import data, pipeline, submission  # noqa: E402

OFFICIAL = os.path.join(ROOT, "evaluation", "results", "baseline_smooth50.csv")


def main():
    labels = pipeline.run(data.load(), tags=None, window=50)
    with tempfile.TemporaryDirectory() as folder:
        path = submission.write(labels, "baseline_check", results_dir=folder)
        same = filecmp.cmp(path, OFFICIAL, shallow=False)
    if same:
        print("OK  o pipeline reproduz a baseline_smooth50 oficial byte a byte.")
        return 0
    print("DIFERENTE  o pipeline não reproduz a baseline_smooth50. Há um bug na base.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
