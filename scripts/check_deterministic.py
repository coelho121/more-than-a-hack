"""Teste de determinismo: gerar a submissão duas vezes tem de dar o mesmo ficheiro, byte a byte.

    python scripts/check_deterministic.py

Corre o scripts/make_submission.py duas vezes, em dois processos separados (como se fossem
duas pessoas a correr o repo), e compara os dois ficheiros pelo SHA-256. É a regra 3 do
desafio: o repositório tem de regenerar exatamente o ficheiro submetido.
"""

import hashlib
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "evaluation", "results")
NAMES = ("_determinismo_a", "_determinismo_b")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    script = os.path.join(ROOT, "scripts", "make_submission.py")
    hashes = []
    try:
        for name in NAMES:
            done = subprocess.run([sys.executable, script, "--name", name], cwd=ROOT,
                                  capture_output=True, text=True)
            if done.returncode != 0:
                print("o make_submission.py falhou:\n{}\n{}".format(done.stdout, done.stderr))
                return 1
            path = os.path.join(RESULTS, name + ".csv")
            hashes.append(sha256(path))
            print("{}  {} bytes  sha256 {}".format(name, os.path.getsize(path), hashes[-1]))
    finally:
        for name in NAMES:
            path = os.path.join(RESULTS, name + ".csv")
            if os.path.isfile(path):
                os.remove(path)

    if hashes[0] == hashes[1]:
        print("OK  as duas corridas deram exatamente o mesmo ficheiro.")
        return 0
    print("DIFERENTE  o pipeline não é determinístico. Há um bug a sério.")
    return 1


if __name__ == "__main__":
    sys.exit(main())