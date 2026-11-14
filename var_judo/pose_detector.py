"""Detecção de pose com MediaPipe Tasks (PoseLandmarker, multi-pessoa).

O wheel do MediaPipe disponível só traz a Tasks API. A vantagem é que
``PoseLandmarker`` com ``num_poses=2`` detecta os DOIS atletas nativamente,
sem precisar recortar o frame por zonas. A identidade (atleta 0 / atleta 1)
é mantida estável por um rastreador simples de centroide mais próximo.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision
    _MP_OK = True
except Exception as _e:  # pragma: no cover
    _MP_OK = False
    _MP_ERRO = _e


MODELO_PADRAO = os.path.join(os.path.dirname(__file__), "models",
                             "pose_landmarker_lite.task")

# Índices dos landmarks do BlazePose (33 pontos).
NARIZ = 0
OMBRO_E, OMBRO_D = 11, 12
COTOVELO_E, COTOVELO_D = 13, 14
PULSO_E, PULSO_D = 15, 16
QUADRIL_E, QUADRIL_D = 23, 24
JOELHO_E, JOELHO_D = 25, 26
TORNOZELO_E, TORNOZELO_D = 27, 28


@dataclass
class AtletaPose:
    """Pose de um atleta num frame, em coordenadas de PIXEL do frame."""

    id: int                                  # identidade rastreada (0/1)
    landmarks: np.ndarray                    # shape (33, 2) em pixels
    visiveis: np.ndarray                     # shape (33,) bool de visibilidade
    centroide: Tuple[float, float]
    bbox: Tuple[int, int, int, int]          # x, y, w, h

    def ponto(self, idx: int) -> Tuple[float, float]:
        return float(self.landmarks[idx, 0]), float(self.landmarks[idx, 1])

    def medio(self, a: int, b: int) -> Tuple[float, float]:
        pa, pb = self.landmarks[a], self.landmarks[b]
        return float((pa[0] + pb[0]) / 2), float((pa[1] + pb[1]) / 2)


class PoseDetector:
    """Um detector por câmera (cada um mantém seu próprio fluxo de tempo)."""

    def __init__(self, num_poses: int = 2, modelo: str = MODELO_PADRAO,
                 confianca_min: float = 0.5):
        if not _MP_OK:
            raise RuntimeError(f"MediaPipe indisponível: {_MP_ERRO}")
        if not os.path.exists(modelo):
            raise FileNotFoundError(
                f"Modelo de pose não encontrado: {modelo}. "
                "Baixe pose_landmarker_lite.task (veja README)."
            )
        opcoes = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=modelo),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=num_poses,
            min_pose_detection_confidence=confianca_min,
            min_tracking_confidence=confianca_min,
            min_pose_presence_confidence=confianca_min,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(opcoes)
        self._t0 = time.time()
        self._ult_ts_ms = -1
        # Rastreamento de identidade: último centroide por id.
        self._ultimos_centroides: Dict[int, Tuple[float, float]] = {}

    # ------------------------------------------------------------------
    def processar(self, frame_bgr: np.ndarray) -> List[AtletaPose]:
        """Recebe um frame BGR e devolve a lista de atletas detectados."""
        h, w = frame_bgr.shape[:2]
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # VIDEO mode exige timestamp crescente em ms.
        ts_ms = int((time.time() - self._t0) * 1000)
        if ts_ms <= self._ult_ts_ms:
            ts_ms = self._ult_ts_ms + 1
        self._ult_ts_ms = ts_ms

        try:
            res = self._landmarker.detect_for_video(mp_img, ts_ms)
        except Exception:
            return []

        atletas_brutos: List[Tuple[np.ndarray, np.ndarray]] = []
        for lms in (res.pose_landmarks or []):
            pts = np.array([[lm.x * w, lm.y * h] for lm in lms], dtype=np.float32)
            vis = np.array([getattr(lm, "visibility", 1.0) > 0.3 for lm in lms],
                           dtype=bool)
            atletas_brutos.append((pts, vis))

        return self._atribuir_identidades(atletas_brutos)

    # ------------------------------------------------------------------
    def _atribuir_identidades(
        self, brutos: List[Tuple[np.ndarray, np.ndarray]]
    ) -> List[AtletaPose]:
        # Calcula centroide de cada detecção.
        candidatos = []
        for pts, vis in brutos:
            c = (float(pts[:, 0].mean()), float(pts[:, 1].mean()))
            candidatos.append((c, pts, vis))

        # Caso simples: ordena por x (esquerda = id 0) quando não há histórico.
        atletas: List[AtletaPose] = []
        usados = set()

        if self._ultimos_centroides:
            # Casa cada detecção ao id anterior mais próximo.
            for cid, c_ant in list(self._ultimos_centroides.items()):
                melhor, melhor_d = None, 1e9
                for i, (c, _, _) in enumerate(candidatos):
                    if i in usados:
                        continue
                    d = (c[0] - c_ant[0]) ** 2 + (c[1] - c_ant[1]) ** 2
                    if d < melhor_d:
                        melhor, melhor_d = i, d
                if melhor is not None:
                    usados.add(melhor)
                    atletas.append(self._montar(cid, candidatos[melhor]))

        # Detecções não casadas recebem novos ids (0,1,...).
        proximo_id = 0
        ids_usados = {a.id for a in atletas}
        for i, cand in enumerate(candidatos):
            if i in usados:
                continue
            while proximo_id in ids_usados:
                proximo_id += 1
            ids_usados.add(proximo_id)
            atletas.append(self._montar(proximo_id, cand))

        # Ordena por id e atualiza histórico de centroides.
        atletas.sort(key=lambda a: a.id)
        self._ultimos_centroides = {a.id: a.centroide for a in atletas}
        return atletas

    @staticmethod
    def _montar(cid: int, cand) -> AtletaPose:
        c, pts, vis = cand
        x0, y0 = pts[:, 0].min(), pts[:, 1].min()
        x1, y1 = pts[:, 0].max(), pts[:, 1].max()
        bbox = (int(x0), int(y0), int(x1 - x0), int(y1 - y0))
        return AtletaPose(id=cid, landmarks=pts, visiveis=vis,
                          centroide=c, bbox=bbox)

    def fechar(self) -> None:
        try:
            self._landmarker.close()
        except Exception:
            pass
