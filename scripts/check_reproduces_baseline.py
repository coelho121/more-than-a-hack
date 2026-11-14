"""Teste de sanidade: a nossa base tem de dar EXATAMENTE a baseline_smooth50 oficial.

    python scripts/check_reproduces_baseline.py

Se der igual: a leitura dos dados, o alinhamento dos frames e a escrita do CSV estão certos.
Se não der: há um bug na base, e tudo o que se construir por cima está torto. Corre isto
sempre que mexeres em src/data.py, src/features.py ou src/submission.py.

A comparação ignora o tipo de fim de linha (\\n vs \\r\\n), porque o git no Windows pode
converter o ficheiro oficial ao fazer checkout. O que tem de ser igual é o conteúdo.
"""

import os
import sys
import tempfile
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import data, pipeline, submission  # noqa: E402

OFFICIAL = os.path.join(ROOT, "evaluation", "results", "baseline_smooth50.csv")


def read_normalised(path):
    """(bytes com fins de linha em \\n, descrição do fim de linha original)."""
    with open(path, "rb") as handle:
        raw = handle.read()
    ending = "CRLF (\\r\\n)" if b"\r\n" in raw else "LF (\\n)"
    return raw.replace(b"\r\n", b"\n"), ending


def main():
    labels = pipeline.baseline(data.load(), window=50)
    with tempfile.TemporaryDirectory() as folder:
        path = submission.write(labels, "baseline_check", results_dir=folder)
        ours, ours_ending = read_normalised(path)
    official, official_ending = read_normalised(OFFICIAL)

    print("fim de linha  nosso: {}   oficial: {}".format(ours_ending, official_ending))

    if ours == official:
        print("OK  o pipeline reproduz a baseline_smooth50 oficial (conteúdo idêntico).")
        return 0

    # Diferença a sério: mostrar só contagens agregadas.
    ours_lines = ours.decode().splitlines()
    official_lines = official.decode().splitlines()
    print("linhas  nosso: {}   oficial: {}".format(len(ours_lines), len(official_lines)))
    pairs = Counter()
    for a, b in zip(ours_lines[1:], official_lines[1:]):
        la, lb = a.split(",")[-1], b.split(",")[-1]
        if la != lb:
            pairs[(la, lb)] += 1
    print("labels diferentes: {}".format(sum(pairs.values())))
    for (la, lb), n in sorted(pairs.items()):
        print("  nosso {} / oficial {}: {}".format(la, lb, n))
    print("DIFERENTE  o pipeline não reproduz a baseline_smooth50. Há um bug na base.")
    return 1


if __name__ == "__main__":
    sys.exit(main())