# More than a Hack 2027 — Fase 1 (posse de bola)

Repo da equipa: Martim, Valido e Gonçalo. **Privado** até à submissão; no fim torna-se público
(é obrigatório entregar o repo público).

Objetivo da fase 1: para cada um dos 199 319 frames do jogo Barça–Athletic, dizer `H` (casa
com bola), `A` (fora com bola) ou `D` (bola morta). Ranking por **spellF1**. Passam as 20
melhores equipas. Prazo: **17 nov 2026**.

Os documentos oficiais estão em [`docs/oficial/`](docs/oficial/). Comecem pelo
[CHALLENGE](docs/oficial/CHALLENGE.md), [EVALUATION](docs/oficial/EVALUATION.md) e
[RULES](docs/oficial/RULES.md).

---

## ⚠️ As 4 regras que não se discutem

1. **Os dados do jogo nunca entram no git.** Nem os ficheiros originais, nem a cache, nem
   submissões. O `.gitignore` já trata disso; não usem `git add -f`.
2. **Os dados nunca vão para um LLM na cloud** (ChatGPT, Claude, Gemini, ...). Os termos proíbem.
   Código, erros e scores podem ir; linhas do tracking, Smart Tagging e frames do vídeo não.
   No Colab, desliguem o Gemini.
3. **Nada feito à mão neste jogo.** As labels saem de código. Ver o vídeo para perceber é ok;
   escrever labels, correções ou constantes ligadas a frames específicos não é.
4. **O GT público (5 min) serve para ver se o pipeline está certo**, não para escolher entre
   versões nem para afinar parâmetros. Também é regra.

---

## Setup (uma vez, cada um)

Python **3.11 ou 3.12**.

```bash
git clone https://github.com/<utilizador>/<repo>.git
cd <repo>
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
nbstripout --install                 # limpa os outputs dos notebooks em cada commit
```

Depois põe os dados que vieram no link da inscrição dentro de `data/`:

```
data/
├─ tactical/     BAR-ATH_tactical.mp4, _atd.xml, _atd.txt, _homography.csv, _smart_tagging.xml
└─ broadcast/    BAR-ATH_broadcast.mp4, _atd.xml, _atd.txt, _homography.csv
```

(Se os tiveres noutro sítio: `export MTAH_DATA_DIR=/caminho/para/a/pasta`.)

Testa que está tudo bem:

```bash
python scripts/check_reproduces_baseline.py   # tem de dizer OK
python scripts/make_submission.py             # gera e valida evaluation/results/candidato.csv
python evaluation/run_evaluation.py           # scoreboard no GT público
```

A primeira vez demora 1–2 minutos porque cria a cache em `cache/`. Depois é rápido.

---

## Estrutura

```
├─ data/                  dados do jogo (fora do git)
├─ cache/                 tracking convertido para .npz (fora do git)
│
├─ games.py               ┐
├─ evaluation/            │ do repo oficial da Metrica, NÃO MEXER
├─ starter/               ┘ (o scorer em evaluation/ é o que decide o ranking)
│
├─ src/                   O NOSSO CÓDIGO: tudo o que entra na submissão
│  ├─ data.py             lê o tracking uma vez e guarda em cache
│  ├─ tagging.py          Smart Tagging -> intervalos em frames, equipa em H/A
│  ├─ features.py         distâncias, velocidades, bola visível, ...
│  ├─ dead_ball.py        frente 1: bola morta (D)
│  ├─ possession.py       frente 2: quem tem a bola e quando muda
│  ├─ pipeline.py         junta tudo -> uma label por frame
│  └─ submission.py       escrever CSV, validar, score no GT público
│
├─ scripts/
│  ├─ make_submission.py            O comando que gera a submissão final
│  └─ check_reproduces_baseline.py  teste de sanidade da base
│
├─ notebooks/
│  ├─ 00_template.ipynb   copiem para a vossa pasta antes de mexer
│  ├─ martim/  valido/  goncalo/
│
└─ docs/
   ├─ technical_doc.md    o doc técnico, escrito ao longo do caminho
   └─ oficial/            documentação oficial do desafio
```

**Estado atual (v0):** o pipeline faz exatamente o mesmo que a baseline oficial (jogador mais
perto da bola + voto por maioria em ±50 frames). É o ponto de partida: cada melhoria entra em
`dead_ball.py` ou `possession.py`.

---

## Como trabalhamos

- **O `main` corre sempre.** Cada um trabalha num branch (`martim/dead-ball`, `valido/transicoes`, ...)
  e junta ao `main` quando funciona e o `check_reproduces_baseline.py` / `make_submission.py` passam.
- **Notebooks só na tua pasta.** Ideia que funciona → passa para `src/`.
- **Pacote novo?** Adiciona ao `requirements.txt` com versão fixa e avisa.
- **No fim de cada sessão:** commit + entrada no log de sessões com o hash do commit.

### Colab

Dá para usar como máquina: clonar o repo (com um token do GitHub, porque é privado), dados na tua
Google Drive e `MTAH_DATA_DIR` a apontar para lá. Os passos estão na 1.ª célula do
`notebooks/00_template.ipynb`. **Gemini desligado.**

---

## Créditos

`games.py`, `evaluation/`, `starter/` e `docs/oficial/` vêm do
[repo oficial](https://github.com/metrica-sports/more-than-a-hack-2027) da Metrica Sports e do
Barça Innovation Hub, licença MIT (ver `LICENSE-metrica-sports`). Os dados do jogo **não** estão
cobertos por essa licença.
