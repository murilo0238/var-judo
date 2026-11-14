"""Heurísticas de detecção de lances de judô a partir das poses.

IMPORTANTE: estas regras são HEURÍSTICAS para chamar a atenção do operador,
não pontuação oficial. Os limiares ficam em ``config.Limiares`` e precisam
ser calibrados com vídeos reais. A decisão final é sempre humana.

Uma instância de ``JudoAnalyzer`` por câmera. A cada frame recebe a lista de
``AtletaPose`` detectados e devolve os alertas disparados.
"""
from __future__ import annotations

import math
import time
from collections import defaultdict, deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from .alert import Alert, AlertType
from .config import Limiares, Quad
from .pose_detector import (AtletaPose, OMBRO_D, OMBRO_E, QUADRIL_D, QUADRIL_E,
                            PULSO_D, PULSO_E, TORNOZELO_D, TORNOZELO_E)


def _angulo_tronco_vertical(ombro_mid: Tuple[float, float],
                            quadril_mid: Tuple[float, float]) -> float:
    """Ângulo (graus) do tronco em relação à vertical. 0 = em pé, 90 = deitado."""
    dx = quadril_mid[0] - ombro_mid[0]
    dy = quadril_mid[1] - ombro_mid[1]
    # Ângulo da componente horizontal vs vertical.
    return abs(math.degrees(math.atan2(dx, dy)))


def _ponto_no_poligono(p: Tuple[float, float], poly: Quad) -> bool:
    if not poly or len(poly) < 3:
        return True  # sem área definida -> considera tudo dentro
    x, y = p
    dentro = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and \
           (x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi):
            dentro = not dentro
        j = i
    return dentro


class _Timer:
    """Mede há quanto tempo uma condição está continuamente ativa."""

    def __init__(self):
        self.inicio: Optional[float] = None

    def atualizar(self, ativo: bool, agora: float) -> float:
        if ativo:
            if self.inicio is None:
                self.inicio = agora
            return agora - self.inicio
        self.inicio = None
        return 0.0


class JudoAnalyzer:
    def __init__(self, indice_camera: int, limiares: Limiares,
                 tatame_quad: Quad, passividade_seg: float,
                 frame_shape: Optional[Tuple[int, int]] = None):
        self.indice = indice_camera
        self.lim = limiares
        self.quad = tatame_quad
        self.passividade_seg = passividade_seg
        self.frame_h = frame_shape[0] if frame_shape else 480
        self.frame_w = frame_shape[1] if frame_shape else 640

        # Histórico de poses por atleta (ts, AtletaPose).
        self._hist: Dict[int, Deque[Tuple[float, AtletaPose]]] = defaultdict(
            lambda: deque(maxlen=60))
        # Timers por (tipo, atleta).
        self._timers: Dict[Tuple[str, int], _Timer] = defaultdict(_Timer)
        # Cooldown: último disparo por (tipo, atleta).
        self._ultimo_disparo: Dict[Tuple[str, int], float] = {}
        # Passividade é por par de atletas.
        self._timer_passividade = _Timer()

    # ------------------------------------------------------------------
    def analisar(self, atletas: List[AtletaPose], frame: np.ndarray,
                 ts: float) -> List[Alert]:
        if frame is not None:
            self.frame_h, self.frame_w = frame.shape[:2]
        for a in atletas:
            self._hist[a.id].append((ts, a))

        alertas: List[Alert] = []
        for a in atletas:
            alertas.extend(self._regras_atleta(a, frame, ts))
        alertas.extend(self._regra_passividade(atletas, frame, ts))
        return alertas

    # ------------------------------------------------------------------
    def _disparavel(self, tipo: AlertType, atleta: int, agora: float) -> bool:
        chave = (tipo.value, atleta)
        ult = self._ultimo_disparo.get(chave, -1e9)
        return (agora - ult) >= self.lim.cooldown_alerta

    def _registrar(self, tipo: AlertType, atleta: int, agora: float) -> None:
        self._ultimo_disparo[(tipo.value, atleta)] = agora

    def _criar(self, tipo: AlertType, a: AtletaPose, frame: np.ndarray,
               ts: float, conf: float) -> Alert:
        cx, cy = a.centroide
        return Alert(
            tipo=tipo, camera=self.indice, atleta=a.id, timestamp=ts,
            confianca_heuristica=conf,
            frame=None if frame is None else frame.copy(),
            alvo_seta=(int(cx), int(cy)),
        )

    # ------------------------------------------------------------------
    def _regras_atleta(self, a: AtletaPose, frame: np.ndarray,
                       ts: float) -> List[Alert]:
        out: List[Alert] = []
        ombro_mid = a.medio(OMBRO_E, OMBRO_D)
        quadril_mid = a.medio(QUADRIL_E, QUADRIL_D)
        angulo = _angulo_tronco_vertical(ombro_mid, quadril_mid)

        # Nível do "solo" do tatame (maior y do polígono ou base do frame).
        if self.quad:
            solo_y = max(p[1] for p in self.quad)
            topo_y = min(p[1] for p in self.quad)
        else:
            solo_y, topo_y = self.frame_h, 0
        faixa = (solo_y - topo_y) * self.lim.faixa_solo
        ombros_no_solo = ombro_mid[1] >= (solo_y - faixa)

        # ---- Ippon: tronco muito inclinado + costas no solo ----
        cond_ippon = (angulo > self.lim.angulo_ippon) and ombros_no_solo
        t = self._timers[("ippon_t", a.id)].atualizar(cond_ippon, ts)
        if t >= self.lim.tempo_ippon and self._disparavel(AlertType.IPPON, a.id, ts):
            self._registrar(AlertType.IPPON, a.id, ts)
            out.append(self._criar(AlertType.IPPON, a, frame, ts,
                                    min(1.0, 0.5 + t / 6)))
            return out  # ippon tem precedência sobre waza-ari

        # ---- Waza-ari: tronco 45-90 + região baixa (contato lateral) ----
        cond_waza = (self.lim.angulo_wazaari_min <= angulo
                     <= self.lim.angulo_wazaari_max) and \
            (quadril_mid[1] >= solo_y - (solo_y - topo_y) * 0.5)
        t = self._timers[("waza_t", a.id)].atualizar(cond_waza, ts)
        if t >= self.lim.tempo_wazaari and self._disparavel(AlertType.WAZA_ARI, a.id, ts):
            self._registrar(AlertType.WAZA_ARI, a.id, ts)
            out.append(self._criar(AlertType.WAZA_ARI, a, frame, ts,
                                    min(1.0, 0.4 + t / 4)))

        # ---- Yoko-guruma: alta velocidade angular + queda ----
        if self._yoko_guruma(a, ts) and self._disparavel(AlertType.YOKO_GURUMA, a.id, ts):
            self._registrar(AlertType.YOKO_GURUMA, a.id, ts)
            out.append(self._criar(AlertType.YOKO_GURUMA, a, frame, ts, 0.5))

        # ---- Shido saída de área: ambos os tornozelos fora ----
        tor_e = a.ponto(TORNOZELO_E)
        tor_d = a.ponto(TORNOZELO_D)
        fora = (not _ponto_no_poligono(tor_e, self.quad)) and \
               (not _ponto_no_poligono(tor_d, self.quad))
        t = self._timers[("saida_t", a.id)].atualizar(fora, ts)
        if t >= self.lim.tempo_saida and self._disparavel(AlertType.SHIDO_SAIDA, a.id, ts):
            self._registrar(AlertType.SHIDO_SAIDA, a.id, ts)
            out.append(self._criar(AlertType.SHIDO_SAIDA, a, frame, ts, 0.7))

        # ---- Shido grip ilegal: pulso abaixo do quadril ----
        margem = abs(quadril_mid[1] - ombro_mid[1]) * 0.3
        pulso_e = a.ponto(PULSO_E)
        pulso_d = a.ponto(PULSO_D)
        mao_baixa = (pulso_e[1] > quadril_mid[1] + margem) or \
                    (pulso_d[1] > quadril_mid[1] + margem)
        t = self._timers[("grip_t", a.id)].atualizar(mao_baixa, ts)
        if t >= self.lim.tempo_grip and self._disparavel(AlertType.SHIDO_GRIP, a.id, ts):
            self._registrar(AlertType.SHIDO_GRIP, a.id, ts)
            out.append(self._criar(AlertType.SHIDO_GRIP, a, frame, ts, 0.5))

        return out

    # ------------------------------------------------------------------
    def _yoko_guruma(self, a: AtletaPose, ts: float) -> bool:
        h = self._hist[a.id]
        if len(h) < 3:
            return False
        janela = [(t, p) for (t, p) in h if ts - t <= 0.6]
        if len(janela) < 3:
            return False
        t0, p0 = janela[0]
        t1, p1 = janela[-1]
        dt = max(1e-3, t1 - t0)
        ang0 = _angulo_tronco_vertical(p0.medio(OMBRO_E, OMBRO_D),
                                       p0.medio(QUADRIL_E, QUADRIL_D))
        ang1 = _angulo_tronco_vertical(p1.medio(OMBRO_E, OMBRO_D),
                                       p1.medio(QUADRIL_E, QUADRIL_D))
        vel_ang = abs(ang1 - ang0) / dt
        queda = (p1.centroide[1] - p0.centroide[1]) / self.frame_h
        return (vel_ang >= self.lim.velocidade_angular_yokoguruma and
                queda >= self.lim.queda_vertical_yokoguruma)

    # ------------------------------------------------------------------
    def _regra_passividade(self, atletas: List[AtletaPose], frame: np.ndarray,
                           ts: float) -> List[Alert]:
        if len(atletas) < 1:
            self._timer_passividade.atualizar(False, ts)
            return []
        # Todos os atletas devem estar com movimento abaixo do limiar.
        todos_parados = True
        for a in atletas:
            h = self._hist[a.id]
            if len(h) < 2:
                todos_parados = False
                break
            (_, ant), (_, atual) = h[-2], h[-1]
            vis = atual.visiveis & ant.visiveis
            if vis.sum() == 0:
                continue
            desloc = np.linalg.norm(
                atual.landmarks[vis] - ant.landmarks[vis], axis=1).mean()
            if desloc / self.frame_h > self.lim.mov_min_passividade:
                todos_parados = False
                break

        t = self._timer_passividade.atualizar(todos_parados, ts)
        if t >= self.passividade_seg and \
                self._disparavel(AlertType.SHIDO_PASSIVIDADE, -1, ts):
            self._registrar(AlertType.SHIDO_PASSIVIDADE, -1, ts)
            a = atletas[0]
            return [self._criar(AlertType.SHIDO_PASSIVIDADE, a, frame, ts, 0.6)]
        return []
