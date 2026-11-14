"""Painel de alerta colorido com ações do operador."""
from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (QFrame, QHBoxLayout, QLabel, QPushButton,
                             QTextEdit, QVBoxLayout)

from ..alert import Alert


class AlertPanel(QFrame):
    """Mostra o alerta atual e os botões de decisão."""

    enviar_ia = pyqtSignal()
    descartar = pyqtSignal()
    confirmar = pyqtSignal()
    replay = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.alerta_atual: Optional[Alert] = None
        self.setFrameShape(QFrame.StyledPanel)
        self._montar()
        self.esconder()

    def _montar(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 10)

        self.lbl_titulo = QLabel("—")
        self.lbl_titulo.setWordWrap(True)
        f = QFont("Arial", 14, QFont.Bold)
        self.lbl_titulo.setFont(f)
        self.lbl_titulo.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.lbl_titulo)

        self.lbl_info = QLabel("")
        self.lbl_info.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.lbl_info)

        botoes = QHBoxLayout()
        self.btn_replay = QPushButton("⟲ Replay do alerta")
        self.btn_ia = QPushButton("🤖 Enviar para IA")
        self.btn_descartar = QPushButton("✖ Descartar")
        self.btn_confirmar = QPushButton("✔ Confirmar e mostrar árbitro")
        self.btn_confirmar.setToolTip("Reservado para tela secundária (futuro)")
        for b in (self.btn_replay, self.btn_ia, self.btn_descartar,
                  self.btn_confirmar):
            b.setMinimumHeight(34)
            botoes.addWidget(b)
        lay.addLayout(botoes)

        self.txt_ia = QTextEdit()
        self.txt_ia.setReadOnly(True)
        self.txt_ia.setMaximumHeight(150)
        self.txt_ia.hide()
        lay.addWidget(self.txt_ia)

        self.btn_replay.clicked.connect(self.replay.emit)
        self.btn_ia.clicked.connect(self.enviar_ia.emit)
        self.btn_descartar.clicked.connect(self.descartar.emit)
        self.btn_confirmar.clicked.connect(self.confirmar.emit)

    # ------------------------------------------------------------------
    def mostrar_alerta(self, alerta: Alert):
        self.alerta_atual = alerta
        r, g, b = alerta.cor
        # Texto escuro sobre amarelo, claro sobre vermelho/laranja.
        cor_texto = "#222" if (r + g + b) > 480 else "#fff"
        self.setStyleSheet(
            f"AlertPanel {{ background:rgb({r},{g},{b}); border-radius:8px; }}"
            f"QLabel {{ color:{cor_texto}; }}"
        )
        self.lbl_titulo.setText(alerta.mensagem)
        cam = alerta.camera + 1
        atleta = "—" if alerta.atleta < 0 else f"Atleta {alerta.atleta + 1}"
        self.lbl_info.setText(
            f"Câmera {cam} · {atleta} · {alerta.hora_formatada} · "
            f"confiança heurística {int(alerta.confianca_heuristica * 100)}%"
        )
        self.txt_ia.hide()
        self.txt_ia.clear()
        self.btn_ia.setEnabled(True)
        self.show()

    def mostrar_processando_ia(self):
        self.txt_ia.show()
        self.txt_ia.setStyleSheet("background:#fff; color:#222;")
        self.txt_ia.setPlainText("Analisando frame com GPT-4o Vision...")
        self.btn_ia.setEnabled(False)

    def mostrar_resposta_ia(self, dados: dict):
        self.btn_ia.setEnabled(True)
        self.txt_ia.show()
        self.txt_ia.setStyleSheet("background:#fff; color:#222;")
        self.txt_ia.setHtml(
            f"<b>Classificação:</b> {dados.get('classificacao','—')}<br>"
            f"<b>Justificativa:</b> {dados.get('justificativa','—')}<br>"
            f"<b>Confiança:</b> {dados.get('confianca','—')}<br>"
            f"<small>Modelo: {dados.get('modelo','?')} · "
            f"tempo de resposta: {dados.get('tempo_s','?')} s</small>"
        )

    def mostrar_erro_ia(self, msg: str):
        self.btn_ia.setEnabled(True)
        self.txt_ia.show()
        self.txt_ia.setStyleSheet("background:#fff; color:#a00;")
        self.txt_ia.setPlainText("Erro: " + msg)

    def esconder(self):
        self.alerta_atual = None
        self.hide()
