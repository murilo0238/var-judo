"""Captura de vídeo com buffer circular dos últimos N segundos.

Cada câmera roda em sua própria QThread. Os frames mais recentes são
emitidos para exibição ao vivo e também guardados (em JPEG, para poupar RAM)
num buffer circular que permite replay instantâneo e câmera lenta.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal


FrameBuf = Tuple[float, bytes]  # (timestamp, jpeg_bytes)


class VideoSource(QThread):
    """Lê frames de uma URL de IP Webcam ou de um arquivo de vídeo."""

    # Emite o frame mais recente (BGR) para exibição ao vivo.
    frame_pronto = pyqtSignal(int, np.ndarray)   # (indice_camera, frame)
    # Emite mudança de status de conexão (texto legível).
    status_mudou = pyqtSignal(int, str)

    def __init__(self, indice: int, fonte: str, fps_alvo: int = 15,
                 buffer_seg: int = 30, loop_arquivo: bool = True, parent=None):
        super().__init__(parent)
        self.indice = indice
        self.fonte = fonte
        self.fps_alvo = max(1, fps_alvo)
        self.buffer_seg = buffer_seg
        self.loop_arquivo = loop_arquivo

        self._rodando = False
        self._cap: Optional[cv2.VideoCapture] = None
        self._eh_arquivo = self._fonte_eh_arquivo(fonte)

        maxlen = self.fps_alvo * self.buffer_seg
        self._buffer: Deque[FrameBuf] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._ultimo_frame: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    @staticmethod
    def _fonte_eh_arquivo(fonte: str) -> bool:
        f = fonte.lower()
        return not (f.startswith("http://") or f.startswith("https://")
                    or f.startswith("rtsp://") or f.isdigit())

    def _abrir(self) -> bool:
        fonte = int(self.fonte) if self.fonte.isdigit() else self.fonte
        self._cap = cv2.VideoCapture(fonte)
        return bool(self._cap and self._cap.isOpened())

    # ------------------------------------------------------------------
    def run(self) -> None:
        self._rodando = True
        intervalo = 1.0 / self.fps_alvo
        self.status_mudou.emit(self.indice, "Conectando...")

        while self._rodando:
            if self._cap is None or not self._cap.isOpened():
                if not self._abrir():
                    self.status_mudou.emit(self.indice, "Falha ao conectar — tentando de novo")
                    self._esperar(1.0)
                    continue
                self.status_mudou.emit(self.indice, "Conectado")

            t0 = time.time()
            ok, frame = self._cap.read()
            if not ok or frame is None:
                if self._eh_arquivo and self.loop_arquivo:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                if self._eh_arquivo:
                    self.status_mudou.emit(self.indice, "Fim do vídeo")
                    break
                # IP webcam caiu: reabrir.
                self.status_mudou.emit(self.indice, "Sinal perdido — reconectando")
                self._cap.release()
                self._cap = None
                self._esperar(0.5)
                continue

            agora = time.time()
            self._guardar(agora, frame)
            with self._lock:
                self._ultimo_frame = frame
            self.frame_pronto.emit(self.indice, frame)

            # Throttle para o fps alvo.
            dt = time.time() - t0
            if dt < intervalo:
                self._esperar(intervalo - dt)

        if self._cap is not None:
            self._cap.release()

    def _esperar(self, segundos: float) -> None:
        # msleep cooperativo que respeita o pedido de parada.
        fim = time.time() + segundos
        while self._rodando and time.time() < fim:
            self.msleep(10)

    # ------------------------------------------------------------------
    def _guardar(self, ts: float, frame: np.ndarray) -> None:
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return
        with self._lock:
            self._buffer.append((ts, buf.tobytes()))

    def ultimo_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._ultimo_frame is None else self._ultimo_frame.copy()

    def get_segment(self, t_ini: float, t_fim: float) -> List[Tuple[float, np.ndarray]]:
        """Retorna (timestamp, frame BGR) do buffer no intervalo pedido."""
        with self._lock:
            itens = [(ts, b) for (ts, b) in self._buffer if t_ini <= ts <= t_fim]
        out: List[Tuple[float, np.ndarray]] = []
        for ts, b in itens:
            arr = np.frombuffer(b, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None:
                out.append((ts, frame))
        return out

    def janela_recente(self, segundos: float) -> List[Tuple[float, np.ndarray]]:
        agora = time.time()
        return self.get_segment(agora - segundos, agora)

    def parar(self) -> None:
        self._rodando = False
        self.wait(2000)
