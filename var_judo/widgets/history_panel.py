"""Histórico cronológico dos alertas da luta atual."""
from __future__ import annotations

from typing import List

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
                             QPushButton, QVBoxLayout, QWidget)

from ..alert import Alert, Decisao


class _ItemHistorico(QWidget):
    def __init__(self, alerta: Alert, ao_rever):
        super().__init__()
        self.alerta = alerta
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 4, 6, 4)
        lay.setSpacing(2)

        r, g, b = alerta.cor
        topo = QLabel(f"● {alerta.tipo.value.upper()}")
        topo.setStyleSheet(f"color:rgb({r},{g},{b}); font-weight:bold;")
        lay.addWidget(topo)

        cam = alerta.camera + 1
        atleta = "—" if alerta.atleta < 0 else f"Atleta {alerta.atleta + 1}"
        lay.addWidget(QLabel(f"{alerta.hora_formatada} · Cam {cam} · {atleta}"))

        linha = QHBoxLayout()
        self.lbl_decisao = QLabel(alerta.decisao.value)
        self._cor_decisao()
        linha.addWidget(self.lbl_decisao)
        linha.addStretch()
        btn = QPushButton("Rever")
        btn.setMaximumWidth(70)
        btn.clicked.connect(lambda: ao_rever(alerta))
        linha.addWidget(btn)
        lay.addLayout(linha)

    def _cor_decisao(self):
        cores = {
            Decisao.PENDENTE: "#caa700",
            Decisao.CONFIRMADO: "#2a8a2a",
            Decisao.DESCARTADO: "#999",
        }
        self.lbl_decisao.setText("Decisão: " + self.alerta.decisao.value)
        self.lbl_decisao.setStyleSheet(
            f"color:{cores.get(self.alerta.decisao, '#999')};")

    def atualizar_decisao(self):
        self._cor_decisao()


class HistoryPanel(QWidget):
    rever = pyqtSignal(object)   # emite o Alert a revisar

    def __init__(self, parent=None):
        super().__init__(parent)
        self._itens: List[_ItemHistorico] = []
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        titulo = QLabel("Histórico da luta")
        titulo.setFont(QFont("Arial", 12, QFont.Bold))
        lay.addWidget(titulo)
        self.lista = QListWidget()
        lay.addWidget(self.lista)
        self.lbl_vazio = QLabel("Nenhum alerta ainda.")
        self.lbl_vazio.setStyleSheet("color:#888;")
        lay.addWidget(self.lbl_vazio)
        self.setMinimumWidth(240)

    def adicionar(self, alerta: Alert):
        self.lbl_vazio.hide()
        widget = _ItemHistorico(alerta, self.rever.emit)
        item = QListWidgetItem(self.lista)
        item.setSizeHint(widget.sizeHint())
        self.lista.addItem(item)
        self.lista.setItemWidget(item, widget)
        self.lista.scrollToBottom()
        self._itens.append(widget)

    def atualizar(self):
        for w in self._itens:
            w.atualizar_decisao()
