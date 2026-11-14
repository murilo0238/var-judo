"""Modelo de alerta: tipos, cores, mensagens em português e dataclass Alert."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

import numpy as np


class AlertType(Enum):
    IPPON = "ippon"
    WAZA_ARI = "waza-ari"
    YOKO_GURUMA = "yoko-guruma"
    SHIDO_PASSIVIDADE = "shido-passividade"
    SHIDO_SAIDA = "shido-saida"
    SHIDO_GRIP = "shido-grip"


class Decisao(Enum):
    PENDENTE = "Pendente"
    CONFIRMADO = "Confirmado"
    DESCARTADO = "Descartado"


# Cor de fundo do painel por tipo (R, G, B).
CORES = {
    AlertType.IPPON: (200, 30, 30),            # vermelho
    AlertType.WAZA_ARI: (230, 120, 20),        # laranja
    AlertType.YOKO_GURUMA: (230, 120, 20),     # laranja
    AlertType.SHIDO_PASSIVIDADE: (220, 190, 20),  # amarelo
    AlertType.SHIDO_SAIDA: (220, 190, 20),
    AlertType.SHIDO_GRIP: (220, 190, 20),
}

# Mensagens em português exibidas no painel de alerta.
MENSAGENS = {
    AlertType.IPPON: "Possível ippon detectado — assistir frame",
    AlertType.WAZA_ARI: "Possível waza-ari detectado — assistir frame",
    AlertType.YOKO_GURUMA: "Possível yoko-guruma detectado — assistir frame",
    AlertType.SHIDO_PASSIVIDADE: "Possível shido detectado — passividade — assistir frame",
    AlertType.SHIDO_SAIDA: "Possível shido detectado — saída de área — assistir frame",
    AlertType.SHIDO_GRIP: "Possível shido detectado — pegada ilegal (mão abaixo da cintura) — assistir frame",
}


def cor_do_tipo(tipo: AlertType) -> Tuple[int, int, int]:
    return CORES.get(tipo, (120, 120, 120))


def mensagem_do_tipo(tipo: AlertType) -> str:
    return MENSAGENS.get(tipo, "Situação suspeita detectada — assistir frame")


@dataclass
class Alert:
    """Um alerta gerado pela detecção automática."""

    tipo: AlertType
    camera: int                       # 0 ou 1
    atleta: int                       # índice do atleta envolvido (0/1) ou -1
    timestamp: float                  # epoch (time.time) do momento do lance
    confianca_heuristica: float       # 0..1 — confiança da heurística, não do mérito
    # Frame congelado (BGR) no momento do alerta.
    frame: Optional[np.ndarray] = None
    # Ponto na imagem para onde a seta aponta (centroide do atleta).
    alvo_seta: Optional[Tuple[int, int]] = None

    decisao: Decisao = Decisao.PENDENTE
    resposta_ia: Optional[dict] = None
    criado_em: float = field(default_factory=time.time)

    @property
    def mensagem(self) -> str:
        return mensagem_do_tipo(self.tipo)

    @property
    def cor(self) -> Tuple[int, int, int]:
        return cor_do_tipo(self.tipo)

    @property
    def hora_formatada(self) -> str:
        return time.strftime("%H:%M:%S", time.localtime(self.timestamp))
