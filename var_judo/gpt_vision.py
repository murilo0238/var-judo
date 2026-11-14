"""Análise de frame suspeito via GPT-4o Vision (API real da OpenAI).

Lê a chave de ``OPENAI_API_KEY``. Roda numa QThread para não travar a UI.
Pede uma resposta estruturada em português e faz parsing tolerante.
"""
from __future__ import annotations

import base64
import os
import re
import time
from typing import Optional

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal


PROMPT_SISTEMA = (
    "Você é um árbitro especialista em judô analisando um único frame de vídeo "
    "de uma luta. Avalie tecnicamente o lance mostrado. Seja objetivo e use "
    "terminologia oficial do judô. Responda SEMPRE em português do Brasil e "
    "EXATAMENTE neste formato, uma informação por linha:\n"
    "Classificação: <ippon | waza-ari | yoko-guruma | shido | nada>\n"
    "Justificativa: <explicação técnica curta em português>\n"
    "Confiança: <alta | média | baixa>"
)

PROMPT_USUARIO = (
    "Analise este frame de uma luta de judô. Houve alguma pontuação "
    "(ippon, waza-ari), uma projeção específica (yoko-guruma) ou uma "
    "infração (shido)? Descreva o que vê — posição dos atletas, ângulo do "
    "tronco, contato com o tatame — e classifique o lance."
)


def _para_data_uri(frame_bgr: np.ndarray, max_lado: int = 1024) -> str:
    h, w = frame_bgr.shape[:2]
    escala = min(1.0, max_lado / max(h, w))
    if escala < 1.0:
        frame_bgr = cv2.resize(frame_bgr, (int(w * escala), int(h * escala)))
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _parse_resposta(texto: str) -> dict:
    def pegar(rotulos):
        for r in rotulos:
            m = re.search(rf"{r}\s*[:\-]\s*(.+)", texto, re.IGNORECASE)
            if m:
                return m.group(1).strip()
        return ""
    return {
        "classificacao": pegar(["Classifica[çc][ãa]o", "Classificacao"]) or "—",
        "justificativa": pegar(["Justificativa", "Justifica[çc][ãa]o"]) or texto.strip(),
        "confianca": pegar(["Confian[çc]a"]) or "—",
        "texto_completo": texto.strip(),
    }


class GPTVisionWorker(QThread):
    """Envia um frame ao GPT-4o Vision e emite o resultado parseado."""

    resultado = pyqtSignal(dict)   # {classificacao, justificativa, confianca, tempo_s, ...}
    erro = pyqtSignal(str)

    def __init__(self, frame_bgr: np.ndarray, modelo: str = "gpt-4o", parent=None):
        super().__init__(parent)
        self.frame = frame_bgr
        self.modelo = modelo

    def run(self) -> None:
        chave = os.environ.get("OPENAI_API_KEY")
        if not chave:
            self.erro.emit(
                "OPENAI_API_KEY não definida. Configure a variável de ambiente "
                "com sua chave da OpenAI e reinicie o programa."
            )
            return
        try:
            from openai import OpenAI
        except Exception as e:  # pragma: no cover
            self.erro.emit(f"Biblioteca openai indisponível: {e}")
            return

        try:
            cliente = OpenAI(api_key=chave)
            data_uri = _para_data_uri(self.frame)
            t0 = time.perf_counter()
            resp = cliente.chat.completions.create(
                model=self.modelo,
                max_tokens=400,
                messages=[
                    {"role": "system", "content": PROMPT_SISTEMA},
                    {"role": "user", "content": [
                        {"type": "text", "text": PROMPT_USUARIO},
                        {"type": "image_url",
                         "image_url": {"url": data_uri, "detail": "high"}},
                    ]},
                ],
            )
            dt = time.perf_counter() - t0
            texto = resp.choices[0].message.content or ""
            dados = _parse_resposta(texto)
            dados["tempo_s"] = round(dt, 2)
            dados["modelo"] = self.modelo
            self.resultado.emit(dados)
        except Exception as e:
            self.erro.emit(f"Falha ao chamar a OpenAI: {e}")
