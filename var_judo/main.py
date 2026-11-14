"""Ponto de entrada: abre a tela de setup e, ao confirmar, o VAR principal."""
from __future__ import annotations

import sys

from PyQt5.QtWidgets import QApplication, QDialog

from .main_window import MainWindow
from .setup_window import SetupWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    setup = SetupWindow()
    if setup.exec_() != QDialog.Accepted or setup.config is None:
        return 0

    janela = MainWindow(setup.config)
    janela.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
