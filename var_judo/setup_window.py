"""Tela de configuração inicial: câmeras, área do tatame, passividade, nome."""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QColor, QImage, QPainter, QPen, QPixmap, QPolygon
from PyQt5.QtWidgets import (QApplication, QDialog, QDoubleSpinBox, QFileDialog,
                             QFormLayout, QGroupBox, QHBoxLayout, QLabel,
                             QLineEdit, QMessageBox, QPushButton, QVBoxLayout)

from .config import AppConfig, Quad


def _capturar_frame(fonte: str, tentativas: int = 30) -> Optional[np.ndarray]:
    """Abre a fonte e lê um frame válido (descarta os primeiros para 'aquecer')."""
    alvo = int(fonte) if fonte.isdigit() else fonte
    cap = cv2.VideoCapture(alvo)
    frame = None
    for _ in range(tentativas):
        ok, f = cap.read()
        if ok and f is not None:
            frame = f
    cap.release()
    return frame


class _DialogoCantos(QDialog):
    """Permite clicar nos 4 cantos do tatame sobre um frame."""

    def __init__(self, frame_bgr: np.ndarray, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Definir área do tatame — clique nos 4 cantos")
        self.frame = frame_bgr
        self.pontos: List[Tuple[int, int]] = []

        h, w = frame_bgr.shape[:2]
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        self._qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        self._iw, self._ih = w, h

        lay = QVBoxLayout(self)
        self.lbl = QLabel()
        self.lbl.setAlignment(Qt.AlignCenter)
        self.lbl.setMinimumSize(640, 480)
        self.lbl.mousePressEvent = self._clique
        lay.addWidget(self.lbl)

        info = QLabel("Clique nos 4 cantos do tatame, em ordem (horário ou anti-horário).")
        info.setStyleSheet("color:#444;")
        lay.addWidget(info)

        botoes = QHBoxLayout()
        b_desfazer = QPushButton("Desfazer")
        b_limpar = QPushButton("Limpar")
        b_ok = QPushButton("OK")
        b_cancel = QPushButton("Cancelar")
        b_desfazer.clicked.connect(self._desfazer)
        b_limpar.clicked.connect(self._limpar)
        b_ok.clicked.connect(self._ok)
        b_cancel.clicked.connect(self.reject)
        for b in (b_desfazer, b_limpar, b_ok, b_cancel):
            botoes.addWidget(b)
        lay.addLayout(botoes)

        self._redesenhar()

    def _mapa(self):
        pm = self.lbl.pixmap()
        if pm is None:
            return 1.0, 0, 0
        lw, lh = self.lbl.width(), self.lbl.height()
        pw, ph = pm.width(), pm.height()
        ox = (lw - pw) // 2
        oy = (lh - ph) // 2
        escala = pw / self._iw
        return escala, ox, oy

    def _clique(self, e):
        if len(self.pontos) >= 4:
            return
        escala, ox, oy = self._mapa()
        ix = (e.pos().x() - ox) / escala
        iy = (e.pos().y() - oy) / escala
        if 0 <= ix <= self._iw and 0 <= iy <= self._ih:
            self.pontos.append((int(ix), int(iy)))
            self._redesenhar()

    def _desfazer(self):
        if self.pontos:
            self.pontos.pop()
            self._redesenhar()

    def _limpar(self):
        self.pontos.clear()
        self._redesenhar()

    def _redesenhar(self):
        disp = QPixmap.fromImage(self._qimg).scaled(
            640, 480, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        escala = disp.width() / self._iw
        p = QPainter(disp)
        pen = QPen(QColor(0, 220, 0), 3)
        p.setPen(pen)
        poly = QPolygon([QPoint(int(x * escala), int(y * escala))
                         for x, y in self.pontos])
        if len(self.pontos) >= 2:
            p.drawPolyline(poly)
        if len(self.pontos) == 4:
            p.drawLine(poly[3], poly[0])
        p.setBrush(QColor(0, 220, 0))
        for x, y in self.pontos:
            p.drawEllipse(QPoint(int(x * escala), int(y * escala)), 5, 5)
        p.end()
        self.lbl.setPixmap(disp)

    def _ok(self):
        if len(self.pontos) != 4:
            QMessageBox.warning(self, "Atenção",
                                "Marque exatamente 4 cantos.")
            return
        self.accept()


class SetupWindow(QDialog):
    """Coleta a configuração da sessão e devolve um AppConfig."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("VAR Judô — Configuração inicial")
        self.config: Optional[AppConfig] = None
        self.quad1: Quad = []
        self.quad2: Quad = []
        self._montar()
        self.resize(560, 420)

    def _montar(self):
        lay = QVBoxLayout(self)

        form = QFormLayout()
        self.ed_nome = QLineEdit("Competição de Judô")
        form.addRow("Nome da competição:", self.ed_nome)

        self.ed_cam1 = QLineEdit("http://192.168.0.101:8080/video")
        self.ed_cam2 = QLineEdit("http://192.168.0.102:8080/video")
        form.addRow("Câmera 1 (URL/vídeo):", self._linha_fonte(self.ed_cam1))
        form.addRow("Câmera 2 (URL/vídeo):", self._linha_fonte(self.ed_cam2))

        self.sp_passividade = QDoubleSpinBox()
        self.sp_passividade.setRange(5, 120)
        self.sp_passividade.setValue(20)
        self.sp_passividade.setSuffix(" s")
        form.addRow("Tempo de passividade:", self.sp_passividade)
        lay.addLayout(form)

        # Área do tatame.
        box = QGroupBox("Área do tatame (clique nos 4 cantos)")
        bl = QHBoxLayout(box)
        self.btn_area1 = QPushButton("Definir área Câmera 1")
        self.btn_area2 = QPushButton("Definir área Câmera 2")
        self.lbl_area1 = QLabel("não definida")
        self.lbl_area2 = QLabel("não definida")
        self.btn_area1.clicked.connect(lambda: self._definir_area(1))
        self.btn_area2.clicked.connect(lambda: self._definir_area(2))
        bl.addWidget(self.btn_area1)
        bl.addWidget(self.lbl_area1)
        bl.addWidget(self.btn_area2)
        bl.addWidget(self.lbl_area2)
        lay.addWidget(box)

        dica = QLabel(
            "Para testar sem celulares, use 'Vídeo…' e escolha um arquivo de "
            "luta gravada. A área do tatame é opcional, mas melhora a detecção "
            "de saída de área."
        )
        dica.setWordWrap(True)
        dica.setStyleSheet("color:#666;")
        lay.addWidget(dica)

        botoes = QHBoxLayout()
        botoes.addStretch()
        b_iniciar = QPushButton("Iniciar VAR")
        b_iniciar.setDefault(True)
        b_iniciar.setMinimumWidth(140)
        b_iniciar.clicked.connect(self._iniciar)
        b_sair = QPushButton("Sair")
        b_sair.clicked.connect(self.reject)
        botoes.addWidget(b_sair)
        botoes.addWidget(b_iniciar)
        lay.addLayout(botoes)

    def _linha_fonte(self, edit: QLineEdit):
        from PyQt5.QtWidgets import QWidget
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(edit)
        b = QPushButton("Vídeo…")
        b.setMaximumWidth(70)
        b.clicked.connect(lambda: self._escolher_video(edit))
        h.addWidget(b)
        return w

    def _escolher_video(self, edit: QLineEdit):
        caminho, _ = QFileDialog.getOpenFileName(
            self, "Escolher vídeo de teste", "",
            "Vídeos (*.mp4 *.avi *.mov *.mkv);;Todos (*.*)")
        if caminho:
            edit.setText(caminho)

    def _definir_area(self, cam: int):
        fonte = self.ed_cam1.text().strip() if cam == 1 else self.ed_cam2.text().strip()
        if not fonte:
            QMessageBox.warning(self, "Atenção", "Informe a fonte da câmera primeiro.")
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        frame = _capturar_frame(fonte)
        QApplication.restoreOverrideCursor()
        if frame is None:
            QMessageBox.critical(self, "Erro",
                                 "Não foi possível capturar um frame dessa fonte.")
            return
        dlg = _DialogoCantos(frame, self)
        if dlg.exec_() == QDialog.Accepted:
            if cam == 1:
                self.quad1 = [list(p) for p in dlg.pontos]
                self.lbl_area1.setText("definida ✓")
            else:
                self.quad2 = [list(p) for p in dlg.pontos]
                self.lbl_area2.setText("definida ✓")

    def _iniciar(self):
        cam1 = self.ed_cam1.text().strip()
        cam2 = self.ed_cam2.text().strip()
        if not cam1 or not cam2:
            QMessageBox.warning(self, "Atenção",
                                "Informe as fontes das duas câmeras.")
            return
        cfg = AppConfig(
            nome_competicao=self.ed_nome.text().strip() or "Competição de Judô",
            cam1_fonte=cam1,
            cam2_fonte=cam2,
            tatame_quad_cam1=self.quad1,
            tatame_quad_cam2=self.quad2,
            passividade_seg=self.sp_passividade.value(),
        )
        cfg.limiares.passividade_seg_padrao = self.sp_passividade.value()
        self.config = cfg
        try:
            cfg.salvar(os.path.join(os.getcwd(), "config_atual.json"))
        except Exception:
            pass
        self.accept()
