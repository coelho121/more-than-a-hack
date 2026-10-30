"""O código da equipa. Tudo o que entra na submissão vive aqui.

Os notebooks experimentam; quando uma ideia funciona, passa para um destes módulos e o
scripts/make_submission.py passa a usá-la.

    data.py        ler o tracking UMA vez e guardar em cache/ (arrays numpy)
    tagging.py     Smart Tagging -> intervalos em frames, com a equipa já em H/A
    features.py    coisas calculadas a partir do tracking (distâncias, velocidades, ...)
    dead_ball.py   frente 1: quando é que a bola está morta (D)
    possession.py  frente 2: quem tem a bola e quando muda (H/A)
    pipeline.py    junta tudo e devolve uma label por frame
    submission.py  escrever o CSV, validar e dar score no GT público
"""

import os
import sys

# Raiz do repo, para conseguirmos importar games.py, evaluation/ e starter/ do repo oficial.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (ROOT, os.path.join(ROOT, "evaluation"), os.path.join(ROOT, "starter")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

CACHE_DIR = os.path.join(ROOT, "cache")
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
