"""Ler o tracking (ATD) uma vez e guardar em cache como arrays numpy.

Porquê: carregar o jogo inteiro com o kloppy demora minutos e gasta gigas de RAM (o Colab
grátis rebenta). Aqui lemos o ficheiro raw linha a linha, uma única vez, e guardamos tudo em
cache/atd_<source>.npz. Daí para a frente, load() demora segundos.

    from src import data
    atd = data.load()          # cria a cache na primeira vez
    atd["ball_xy"]             # (N, 2)    bola, coordenadas 0-1, NaN quando não se vê
    atd["players_xy"]          # (N, S, 2) cada track slot, coordenadas 0-1, NaN quando não existe
    atd["players_s"]           # (N, S)    velocidade de cada slot em m/s
    atd["team_of_track"]       # (S,)      0 = casa (Barcelona, H), 1 = fora (Athletic, A)

O índice da 1.ª dimensão É o frame do vídeo, ou seja, a linha da submissão (0 ... 199318).
A conversão frame_id - 1 já está feita aqui, ninguém precisa de pensar nisso outra vez.

ATENÇÃO: a cache deriva dos dados do jogo. Fica em cache/, que está no .gitignore. Nunca a
partilhem fora da equipa nem a colem num LLM.
"""

import os
import xml.etree.ElementTree as ET

import numpy as np

from src import CACHE_DIR
from games import GAME, GAMES, require, source_frames

# Formato de cada linha do ATD raw:  "<frame_id>:<slot>;<slot>;...:<bola>"
# cada grupo é "x,y,velocidade", com NaN quando não há observação.
FIELD_SEP = ":"
GROUP_SEP = ";"
VALUE_SEP = ","

HOME, AWAY = 0, 1


def cache_path(source="tactical"):
    return os.path.join(CACHE_DIR, "atd_{}.npz".format(source))


def team_of_each_track(meta_path):
    """(S,) array: 0 se o slot é da equipa da casa, 1 se é de fora.

    A primeira equipa no XML é a da casa (convenção FIFA/EPTS, a mesma que o baseline usa).
    Um slot NÃO é um jogador: há 46 slots para 22 jogadores e trocam de identidade.
    """
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("{}: esperava 2 equipas, encontrei {}".format(meta_path, len(team_ids)))
    slots = [player.get("teamId") for player in root.iter("Player")]
    return np.array([HOME if t == team_ids[0] else AWAY for t in slots], dtype=np.int8)


def team_names(meta_path):
    root = ET.parse(meta_path).getroot()
    return tuple(team.findtext("Name") for team in root.iter("Team"))


def _parse_group_list(text, n_groups):
    """'x,y,s;x,y,s;...' -> array (n_groups, 3). Grupos em falta ficam NaN."""
    values = np.array(text.replace(GROUP_SEP, VALUE_SEP).split(VALUE_SEP), dtype=np.float64)
    out = np.full((n_groups, 3), np.nan, dtype=np.float64)
    usable = min(n_groups, len(values) // 3)
    out[:usable] = values[:usable * 3].reshape(usable, 3)
    return out


def build_cache(source="tactical", game=GAME, data_dir=None, force=False, verbose=True):
    """Lê o ATD raw e escreve cache/atd_<source>.npz. Devolve o caminho."""
    path = cache_path(source)
    if os.path.isfile(path) and not force:
        return path

    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir, source)
    team_of_track = team_of_each_track(meta_path)
    n_slots = len(team_of_track)
    n_frames = source_frames(source, game)

    ball = np.full((n_frames, 3), np.nan, dtype=np.float64)
    players = np.full((n_frames, n_slots, 3), np.nan, dtype=np.float64)

    if verbose:
        print("a ler {} ({} frames, {} slots) ...".format(
            os.path.basename(raw_path), n_frames, n_slots))
    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(FIELD_SEP)
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1          # frame_id conta de 1, o vídeo de 0
            if not 0 <= frame < n_frames:
                continue
            players[frame] = _parse_group_list(parts[1], n_slots)
            ball[frame] = _parse_group_list(parts[2], 1)[0]

    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(path,
             ball_xy=ball[:, :2], ball_s=ball[:, 2],
             players_xy=players[:, :, :2], players_s=players[:, :, 2],
             team_of_track=team_of_track,
             team_names=np.array(team_names(meta_path)),
             pitch=np.array(GAMES[game]["pitch"], dtype=np.float64))
    if verbose:
        seen = np.isfinite(ball[:, 0]).mean()
        print("cache escrita em {}  (bola visível em {:.1f}% dos frames)".format(
            os.path.relpath(path), 100 * seen))
    return path


def load(source="tactical", game=GAME, data_dir=None):
    """A cache como dict de arrays. Cria-a se ainda não existir."""
    path = build_cache(source, game, data_dir)
    with np.load(path) as archive:
        return {key: archive[key] for key in archive.files}


if __name__ == "__main__":
    atd = load()
    print({key: value.shape for key, value in atd.items()})
