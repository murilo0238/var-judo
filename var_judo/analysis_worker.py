"""Worker que roda pose + heurísticas fora da thread da UI, por câmera."""
from __future__ import annotations

import time

from PyQt5.QtCore import QThread, pyqtSignal

from .config import AppConfig
from .judo_analyzer import JudoAnalyzer
from .pose_detector import PoseDetector
from .video_source import VideoSource


class AnalysisWorker(QThread):
    alerta = pyqtSignal(object)            # Alert
    poses_prontas = pyqtSignal(int, list)  # (camera, [AtletaPose])
    falhou = pyqtSignal(int, str)

    def __init__(self, indice: int, fonte: VideoSource, config: AppConfig,
                 parent=None):
        super().__init__(parent)
        self.indice = indice
        self.fonte = fonte
        self.config = config
        self._rodando = False

    def run(self) -> None:
        self._rodando = True
        try:
            detector = PoseDetector(
                num_poses=2,
                confianca_min=self.config.limiares.confianca_pose_min)
        except Exception as e:
            self.falhou.emit(self.indice, str(e))
            return

        analyzer = JudoAnalyzer(
            self.indice, self.config.limiares,
            self.config.quad_da_camera(self.indice),
            self.config.passividade_seg)

        # Intervalo alvo entre análises (alivia CPU).
        fps_analise = max(1.0, self.config.fps_alvo / max(1, self.config.pose_a_cada_n))
        intervalo = 1.0 / fps_analise

        while self._rodando:
            t0 = time.time()
            frame = self.fonte.ultimo_frame()
            if frame is None:
                self.msleep(40)
                continue
            try:
                atletas = detector.processar(frame)
                ts = time.time()
                for al in analyzer.analisar(atletas, frame, ts):
                    self.alerta.emit(al)
                self.poses_prontas.emit(self.indice, atletas)
            except Exception as e:
                self.falhou.emit(self.indice, str(e))
                self.msleep(200)

            dt = time.time() - t0
            if dt < intervalo:
                self.msleep(int((intervalo - dt) * 1000))

        detector.fechar()

    def parar(self) -> None:
        self._rodando = False
        self.wait(2000)
