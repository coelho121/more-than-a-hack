"""O comando único que gera a submissão a partir dos dados. É isto que os organizadores correm.

    python scripts/make_submission.py                      # gera evaluation/results/candidato.csv
    python scripts/make_submission.py --name teste_v2      # outro nome
    python scripts/make_submission.py --window 25          # outro parâmetro

O que faz, por ordem:
  1. lê o tracking (cria a cache em cache/ na primeira vez, demora 1-2 min)
  2. lê o Smart Tagging
  3. corre o pipeline (src/pipeline.py)
  4. escreve o CSV em evaluation/results/
  5. valida com as mesmas regras do scorer oficial
  6. mostra o score no GT público (SÓ para ver se nada está partido, nunca para escolher versões)

Determinístico: o mesmo código + os mesmos dados = o mesmo ficheiro, byte a byte. Se algum dia
isto deixar de ser verdade (aleatoriedade, ordem de dicionários, versões), é um bug a sério.
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import data, pipeline, submission, tagging  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default="candidato", help="nome do ficheiro, sem .csv")
    parser.add_argument("--window", type=int, default=50,
                        help="meia-janela do voto por maioria, em frames (v0 = baseline)")
    parser.add_argument("--no-tags", action="store_true", help="não ler o Smart Tagging")
    args = parser.parse_args()

    atd = data.load()
    tags = None if args.no_tags else tagging.load()

    labels = pipeline.run(atd, tags, window=args.window)
    path = submission.write(labels, args.name)

    reason = submission.validate(path)
    if reason:
        print("FALHOU a validação: {}".format(reason))
        return 1
    print("OK  {}".format(os.path.relpath(path, ROOT)))
    print("GT público (só verificação, NÃO usar para escolher versões):",
          submission.score_public(labels))
    return 0


if __name__ == "__main__":
    sys.exit(main())
