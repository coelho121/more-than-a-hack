"""Frente 1: quando é que a bola está morta (D).

Estado atual (v0): NÃO faz nada próprio. O baseline só diz D quando o tracking perde a bola
durante uma janela inteira, por isso a classe D dele é fraca. É aqui que isso melhora.

Ideias para começar (ver docs/oficial/DATA.md e o notebook 01):
  - Smart Tagging: cada SET PIECE é um reinício, logo acaba uma paragem. Atenção que o
    intervalo começa ~3.5 s antes do reinício real.
  - Bola fora do campo (coordenadas fora de 0-1) ou parada muito tempo.
  - Jogadores todos lentos ao mesmo tempo (paragem de jogo).
  - Lembrete: a 2.ª parte é avaliada desde o frame 110808 mas só começa no 113742.
    Esses frames são todos D.

Contrato: dead_ball_mask(atd, tags) devolve um array bool (N,) com True onde achamos que a bola
está morta, ou None se esta frente ainda não estiver a ser usada.
"""


def dead_ball_mask(atd, tags=None):
    return None
