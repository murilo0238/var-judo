"""Janela principal do VAR: orquestra câmeras, análise, alertas e replay."""
from __future__ import annotations

import time
from typing import List, Optional, Tuple

import numpy as np
from PyQt5.QtCore import Qt, QTimer, pyqtSlot
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (QButtonGroup, QComboBox, QHBoxLayout, QLabel,
                             QMainWindow, QMessageBox, QPushButton, QSplitter,
                             QVBoxLayout, QWidget)

from .alert import Alert, Decisao
from .analysis_worker import AnalysisWorker
from .config import AppConfig
from .gpt_vision import GPTVisionWorker
from .video_source import VideoSource
from .widgets.alert_panel import AlertPanel
from .widgets.history_panel import HistoryPanel
from .widgets.video_widget import VideoWidget

# Segundos antes/depois do alerta exibidos no replay.
REPLAY_ANTES = 4.0
REPLAY_DEPOIS = 1.5


class MainWindow(QMainWindow):
    def __init__(self, config: AppConfig):
        super().__init__()
        self.config = config
        self.setWindowTitle(f"VAR Judô — {config.nome_competicao}")
        self.resize(1280, 820)

        self._pausado = False
        self._velocidade = 1.0
        self._ult_frame_vivo: List[Optional[np.ndarray]] = [None, None]
        self._alerta_ativo: Optional[Alert] = None
        self._ia_worker: Optional[GPTVisionWorker] = None

        # Replay.
        self._replay_frames: List[np.ndarray] = []
        self._replay_idx = 0
        self._replay_cam = 0
        self._replay_timer = QTimer(self)
        self._replay_timer.timeout.connect(self._replay_tick)

        self._montar_ui()
        self._iniciar_pipeline()

    # ------------------------------------------------------------------
    def _montar_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        raiz = QVBoxLayout(central)

        # Topo: nome da competição.
        topo = QLabel(self.config.nome_competicao)
        topo.setAlignment(Qt.AlignCenter)
        topo.setFont(QFont("Arial", 18, QFont.Bold))
        topo.setStyleSheet("background:#1d2733; color:#fff; padding:8px;")
        raiz.addWidget(topo)

        # Vídeos + histórico.
        split = QSplitter(Qt.Horizontal)
        videos = QWidget()
        vlay = QHBoxLayout(videos)
        vlay.setContentsMargins(0, 0, 0, 0)
        self.video1 = VideoWidget("Câmera 1")
        self.video2 = VideoWidget("Câmera 2")
        vlay.addWidget(self.video1)
        vlay.addWidget(self.video2)
        split.addWidget(videos)

        self.historico = HistoryPanel()
        self.historico.rever.connect(self._rever_alerta)
        split.addWidget(self.historico)
        split.setStretchFactor(0, 4)
        split.setStretchFactor(1, 1)
        raiz.addWidget(split, stretch=1)

        # Controles.
        raiz.addLayout(self._barra_controles())

        # Painel de alerta.
        self.painel = AlertPanel()
        self.painel.enviar_ia.connect(self._enviar_ia)
        self.painel.descartar.connect(self._descartar)
        self.painel.confirmar.connect(self._confirmar)
        self.painel.replay.connect(self._replay_alerta_ativo)
        raiz.addWidget(self.painel)

    def _barra_controles(self):
        barra = QHBoxLayout()
        self.btn_pausa = QPushButton("⏸ Pausar")
        self.btn_pausa.clicked.connect(self._alternar_pausa)
        barra.addWidget(self.btn_pausa)

        barra.addWidget(QLabel("Velocidade:"))
        self.cb_vel = QComboBox()
        self.cb_vel.addItems(["0.25x", "0.5x", "1x"])
        self.cb_vel.setCurrentText("1x")
        self.cb_vel.currentTextChanged.connect(self._mudar_velocidade)
        barra.addWidget(self.cb_vel)

        barra.addWidget(QLabel("Zoom:"))
        b_mais = QPushButton("＋")
        b_menos = QPushButton("－")
        b_reset = QPushButton("Reset")
        b_mais.clicked.connect(lambda: self._zoom(1.25))
        b_menos.clicked.connect(lambda: self._zoom(0.8))
        b_reset.clicked.connect(self._zoom_reset)
        for b in (b_mais, b_menos, b_reset):
            b.setMaximumWidth(70)
            barra.addWidget(b)

        barra.addStretch()
        self.lbl_status = QLabel("Ao vivo")
        self.lbl_status.setStyleSheet("color:#2a8a2a; font-weight:bold;")
        barra.addWidget(self.lbl_status)
        return barra

    # ------------------------------------------------------------------
    def _iniciar_pipeline(self):
        self.fontes: List[VideoSource] = []
        self.workers: List[AnalysisWorker] = []
        for i, fonte in enumerate([self.config.cam1_fonte, self.config.cam2_fonte]):
            vs = VideoSource(i, fonte, self.config.fps_alvo, self.config.buffer_seg)
            vs.frame_pronto.connect(self._on_frame)
            vs.status_mudou.connect(self._on_status)
            vs.start()
            self.fontes.append(vs)

            w = AnalysisWorker(i, vs, self.config)
            w.alerta.connect(self._on_alerta)
            w.falhou.connect(self._on_falha_analise)
            w.start()
            self.workers.append(w)

    # ---- vídeo ao vivo ----
    @pyqtSlot(int, np.ndarray)
    def _on_frame(self, cam: int, frame: np.ndarray):
        self._ult_frame_vivo[cam] = frame
        if self._pausado or self._replay_timer.isActive():
            return
        (self.video1 if cam == 0 else self.video2).set_frame(frame)

    @pyqtSlot(int, str)
    def _on_status(self, cam: int, texto: str):
        (self.video1 if cam == 0 else self.video2).set_status(texto)

    @pyqtSlot(int, str)
    def _on_falha_analise(self, cam: int, msg: str):
        self.statusBar().showMessage(f"Câmera {cam + 1}: análise indisponível — {msg}", 8000)

    # ---- controles ----
    def _alternar_pausa(self):
        self._pausado = not self._pausado
        self.btn_pausa.setText("▶ Retomar" if self._pausado else "⏸ Pausar")
        self._atualizar_status()
        if not self._pausado:
            self.video1.limpar_seta()
            self.video2.limpar_seta()

    def _mudar_velocidade(self, txt: str):
        self._velocidade = {"0.25x": 0.25, "0.5x": 0.5, "1x": 1.0}.get(txt, 1.0)
        if self._replay_timer.isActive():
            self._replay_timer.setInterval(self._intervalo_replay())

    def _zoom(self, fator: float):
        for v in (self.video1, self.video2):
            v.set_zoom(v.zoom() * fator)

    def _zoom_reset(self):
        self.video1.reset_view()
        self.video2.reset_view()

    def _atualizar_status(self):
        if self._replay_timer.isActive():
            self.lbl_status.setText("Replay")
            self.lbl_status.setStyleSheet("color:#1d6fd6; font-weight:bold;")
        elif self._pausado:
            self.lbl_status.setText("Pausado")
            self.lbl_status.setStyleSheet("color:#caa700; font-weight:bold;")
        else:
            self.lbl_status.setText("Ao vivo")
            self.lbl_status.setStyleSheet("color:#2a8a2a; font-weight:bold;")

    # ---- alertas ----
    @pyqtSlot(object)
    def _on_alerta(self, alerta: Alert):
        self.historico.adicionar(alerta)
        # Só assume a tela se não houver alerta sendo tratado.
        if self._alerta_ativo is not None:
            self.statusBar().showMessage(
                f"Novo alerta em fila: {alerta.tipo.value} (ver histórico)", 5000)
            return
        self._exibir_alerta(alerta)

    def _exibir_alerta(self, alerta: Alert):
        self._alerta_ativo = alerta
        self._pausado = True
        self.btn_pausa.setText("▶ Retomar")
        self._atualizar_status()
        vw = self.video1 if alerta.camera == 0 else self.video2
        if alerta.frame is not None:
            vw.set_frame(alerta.frame)
        if alerta.alvo_seta is not None:
            vw.set_seta(alerta.alvo_seta, alerta.cor)
        self.painel.mostrar_alerta(alerta)

    def _enviar_ia(self):
        al = self._alerta_ativo
        if al is None or al.frame is None:
            return
        self.painel.mostrar_processando_ia()
        self._ia_worker = GPTVisionWorker(al.frame, self.config.modelo_ia)
        self._ia_worker.resultado.connect(self._ia_resultado)
        self._ia_worker.erro.connect(self.painel.mostrar_erro_ia)
        self._ia_worker.start()

    @pyqtSlot(dict)
    def _ia_resultado(self, dados: dict):
        if self._alerta_ativo is not None:
            self._alerta_ativo.resposta_ia = dados
        self.painel.mostrar_resposta_ia(dados)

    def _descartar(self):
        if self._alerta_ativo:
            self._alerta_ativo.decisao = Decisao.DESCARTADO
        self._encerrar_alerta()

    def _confirmar(self):
        if self._alerta_ativo:
            self._alerta_ativo.decisao = Decisao.CONFIRMADO
        QMessageBox.information(
            self, "Confirmado",
            "Alerta confirmado. A exibição em tela secundária para o árbitro "
            "será implementada em versão futura.")
        self._encerrar_alerta()

    def _encerrar_alerta(self):
        self.historico.atualizar()
        self._alerta_ativo = None
        self.painel.esconder()
        self.video1.limpar_seta()
        self.video2.limpar_seta()
        # Continua pausado; operador retoma quando quiser.

    # ---- replay ----
    def _intervalo_replay(self) -> int:
        base = 1.0 / max(1, self.config.fps_alvo)
        return max(15, int(base / self._velocidade * 1000))

    def _replay_alerta_ativo(self):
        if self._alerta_ativo:
            self._iniciar_replay(self._alerta_ativo)

    def _rever_alerta(self, alerta: Alert):
        self._iniciar_replay(alerta)

    def _iniciar_replay(self, alerta: Alert):
        cam = alerta.camera
        segmento = self.fontes[cam].get_segment(
            alerta.timestamp - REPLAY_ANTES, alerta.timestamp + REPLAY_DEPOIS)
        self._replay_cam = cam
        vw = self.video1 if cam == 0 else self.video2
        if not segmento:
            # Já saiu do buffer: mostra o frame congelado do alerta.
            if alerta.frame is not None:
                vw.set_frame(alerta.frame)
                if alerta.alvo_seta:
                    vw.set_seta(alerta.alvo_seta, alerta.cor)
            self.statusBar().showMessage(
                "Segmento fora do buffer — exibindo frame do alerta.", 4000)
            return
        self._replay_frames = [f for (_, f) in segmento]
        self._replay_idx = 0
        self._pausado = True
        self._replay_timer.start(self._intervalo_replay())
        self._atualizar_status()

    def _replay_tick(self):
        if self._replay_idx >= len(self._replay_frames):
            self._replay_timer.stop()
            self._atualizar_status()
            return
        frame = self._replay_frames[self._replay_idx]
        vw = self.video1 if self._replay_cam == 0 else self.video2
        vw.set_frame(frame)
        self._replay_idx += 1

    # ------------------------------------------------------------------
    def closeEvent(self, e):
        try:
            self._replay_timer.stop()
            for w in getattr(self, "workers", []):
                w.parar()
            for f in getattr(self, "fontes", []):
                f.parar()
        finally:
            e.accept()
