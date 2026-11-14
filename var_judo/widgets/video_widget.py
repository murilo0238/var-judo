"""Widget de exibição de vídeo com zoom digital, pan e overlay de seta."""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
from PyQt5.QtCore import QPoint, QPointF, QRectF, Qt
from PyQt5.QtGui import (QColor, QFont, QImage, QPainter, QPen, QPixmap,
                         QPolygonF)
from PyQt5.QtWidgets import QWidget


class VideoWidget(QWidget):
    """Mostra frames BGR (numpy) com zoom/pan e uma seta de destaque."""

    def __init__(self, titulo: str = "", parent=None):
        super().__init__(parent)
        self.titulo = titulo
        self.status = ""
        self._qimg: Optional[QImage] = None
        self._img_wh: Tuple[int, int] = (0, 0)
        self._zoom = 1.0
        self._pan = QPointF(0, 0)
        self._arrastando = False
        self._ult_mouse = QPoint()
        self._seta: Optional[Tuple[int, int]] = None
        self._seta_cor = QColor(255, 0, 0)
        self.setMinimumSize(360, 270)
        self.setStyleSheet("background:#111;")
        self.setMouseTracking(True)

    # ------------------------------------------------------------------
    def set_frame(self, frame_bgr: np.ndarray) -> None:
        if frame_bgr is None:
            return
        h, w = frame_bgr.shape[:2]
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        self._qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        self._img_wh = (w, h)
        self.update()

    def set_status(self, texto: str) -> None:
        self.status = texto
        self.update()

    def set_seta(self, alvo: Optional[Tuple[int, int]],
                 cor: Tuple[int, int, int] = (255, 0, 0)) -> None:
        self._seta = alvo
        self._seta_cor = QColor(*cor)
        self.update()

    def limpar_seta(self) -> None:
        self._seta = None
        self.update()

    # ---- zoom / pan ----
    def set_zoom(self, z: float) -> None:
        self._zoom = max(1.0, min(8.0, z))
        if self._zoom == 1.0:
            self._pan = QPointF(0, 0)
        self.update()

    def zoom(self) -> float:
        return self._zoom

    def reset_view(self) -> None:
        self._zoom = 1.0
        self._pan = QPointF(0, 0)
        self.update()

    def wheelEvent(self, e):
        passo = 1.15 if e.angleDelta().y() > 0 else 1 / 1.15
        self.set_zoom(self._zoom * passo)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton and self._zoom > 1.0:
            self._arrastando = True
            self._ult_mouse = e.pos()

    def mouseMoveEvent(self, e):
        if self._arrastando:
            d = e.pos() - self._ult_mouse
            self._pan += QPointF(d)
            self._ult_mouse = e.pos()
            self.update()

    def mouseReleaseEvent(self, e):
        self._arrastando = False

    # ------------------------------------------------------------------
    def _transform(self):
        """Retorna (escala_efetiva, offset_x, offset_y) img->widget."""
        iw, ih = self._img_wh
        if iw == 0:
            return 1.0, 0.0, 0.0
        base = min(self.width() / iw, self.height() / ih)
        eff = base * self._zoom
        ox = (self.width() - iw * eff) / 2 + self._pan.x()
        oy = (self.height() - ih * eff) / 2 + self._pan.y()
        return eff, ox, oy

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(17, 17, 17))

        if self._qimg is None:
            p.setPen(QColor(160, 160, 160))
            p.setFont(QFont("Arial", 12))
            p.drawText(self.rect(), Qt.AlignCenter,
                       f"{self.titulo}\n(sem sinal)")
            return

        eff, ox, oy = self._transform()
        iw, ih = self._img_wh
        alvo = QRectF(ox, oy, iw * eff, ih * eff)
        p.setRenderHint(QPainter.SmoothPixmapTransform, True)
        p.drawImage(alvo, self._qimg)

        # Seta de destaque.
        if self._seta is not None:
            ax = ox + self._seta[0] * eff
            ay = oy + self._seta[1] * eff
            self._desenhar_seta(p, ax, ay)

        # Título e status.
        p.setFont(QFont("Arial", 11, QFont.Bold))
        if self.titulo:
            self._texto_sombra(p, 8, 20, self.titulo, QColor(255, 255, 255))
        if self.status:
            self._texto_sombra(p, 8, self.height() - 10, self.status,
                               QColor(255, 230, 120))

    def _desenhar_seta(self, p: QPainter, ax: float, ay: float):
        # Seta vinda de cima-esquerda apontando para (ax, ay).
        comp = 70
        ang = math.radians(55)
        bx = ax - comp * math.cos(ang)
        by = ay - comp * math.sin(ang)
        pen = QPen(self._seta_cor, 5)
        pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        p.drawLine(int(bx), int(by), int(ax), int(ay))
        # Ponta.
        cabeca = 18
        a1 = math.atan2(ay - by, ax - bx)
        pts = QPolygonF([
            QPointF(ax, ay),
            QPointF(ax - cabeca * math.cos(a1 - 0.4),
                    ay - cabeca * math.sin(a1 - 0.4)),
            QPointF(ax - cabeca * math.cos(a1 + 0.4),
                    ay - cabeca * math.sin(a1 + 0.4)),
        ])
        p.setBrush(self._seta_cor)
        p.drawPolygon(pts)

    @staticmethod
    def _texto_sombra(p: QPainter, x: int, y: int, txt: str, cor: QColor):
        p.setPen(QColor(0, 0, 0))
        p.drawText(x + 1, y + 1, txt)
        p.setPen(cor)
        p.drawText(x, y, txt)
