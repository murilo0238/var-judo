"""Configuração da aplicação e limiares de detecção.

Todos os limiares ficam centralizados em ``Limiares`` para facilitar a
calibração com vídeos reais antes da competição. A detecção automática é
uma HEURÍSTICA de chamar atenção do operador, não um juiz — ajuste estes
valores assistindo lances gravados.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple

# Um ponto (x, y) em coordenadas de pixel do frame.
Ponto = Tuple[float, float]
# Quadrilátero do tatame: 4 cantos.
Quad = List[Ponto]


@dataclass
class Limiares:
    """Limiares de detecção. Edite e recarregue para calibrar."""

    # Ângulo do tronco em relação à vertical (graus).
    angulo_ippon: float = 45.0          # tronco mais inclinado que isto -> candidato a ippon
    angulo_wazaari_min: float = 45.0
    angulo_wazaari_max: float = 90.0

    # Tempos sustentados (segundos).
    tempo_ippon: float = 2.0            # contato nas costas por mais de 2s
    tempo_wazaari: float = 1.0          # contato lateral por mais de 1s
    tempo_grip: float = 3.0             # mão abaixo da cintura por mais de 3s
    tempo_saida: float = 0.5            # ambos os pés fora da área

    # Passividade: deslocamento médio dos landmarks (fração da altura do frame)
    # abaixo do qual o atleta é considerado parado.
    mov_min_passividade: float = 0.012
    passividade_seg_padrao: float = 20.0

    # Yoko-guruma: velocidade angular do tronco (graus/seg) + queda vertical.
    velocidade_angular_yokoguruma: float = 120.0
    queda_vertical_yokoguruma: float = 0.15   # fração da altura do frame

    # "Contato com o solo": fração da altura do tatame a partir da base
    # dentro da qual os ombros são considerados próximos ao chão.
    faixa_solo: float = 0.25

    # Cooldown entre alertas do mesmo tipo/atleta (segundos).
    cooldown_alerta: float = 5.0

    # Confiança mínima da pose (MediaPipe) para considerar um atleta.
    confianca_pose_min: float = 0.5


@dataclass
class AppConfig:
    """Configuração de uma sessão de VAR, definida na tela de setup."""

    nome_competicao: str = "Competição de Judô"
    # Fonte pode ser URL de IP Webcam (http://IP:8080/video) ou caminho de vídeo.
    cam1_fonte: str = ""
    cam2_fonte: str = ""
    # Polígono do tatame por câmera (4 cantos em pixels). Vazio = tela inteira.
    tatame_quad_cam1: Quad = field(default_factory=list)
    tatame_quad_cam2: Quad = field(default_factory=list)

    passividade_seg: float = 20.0
    fps_alvo: int = 15
    buffer_seg: int = 30
    modelo_ia: str = "gpt-4o"

    # Processa pose a cada N frames (alivia CPU). 1 = todo frame.
    pose_a_cada_n: int = 2

    limiares: Limiares = field(default_factory=Limiares)

    # ---- persistência ----
    def salvar(self, caminho: str) -> None:
        with open(caminho, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2)

    @classmethod
    def carregar(cls, caminho: str) -> "AppConfig":
        with open(caminho, "r", encoding="utf-8") as f:
            dados = json.load(f)
        lim = Limiares(**dados.pop("limiares", {}))
        cfg = cls(**dados)
        cfg.limiares = lim
        return cfg

    def quad_da_camera(self, indice: int) -> Quad:
        return self.tatame_quad_cam1 if indice == 0 else self.tatame_quad_cam2
