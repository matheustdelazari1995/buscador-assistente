"""Integração WhatsApp via Evolution API.

Recebe webhooks da Evolution (quando o usuário manda mensagem no WhatsApp),
persiste histórico por número de telefone em SQLite, aciona o agente Claude
e envia a resposta de volta pela Evolution API.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Optional

import aiohttp


# ===========================================================================
# Config (via env)
# ===========================================================================


EVOLUTION_URL = os.getenv("EVOLUTION_URL", "https://evotripse.tripse.com.br").rstrip("/")
EVOLUTION_INSTANCE = os.getenv("EVOLUTION_INSTANCE", "clarisse")
EVOLUTION_API_KEY = os.getenv("EVOLUTION_API_KEY", "")
WEBHOOK_SECRET = os.getenv("WHATSAPP_WEBHOOK_SECRET", "change-me")

# Whitelist: CSV de números (ex: "5527999999999,5521988888888"). Vazio = qualquer um.
_WHITELIST_RAW = os.getenv("WHATSAPP_WHITELIST", "").strip()
WHITELIST: Optional[set[str]] = (
    {n.strip() for n in _WHITELIST_RAW.split(",") if n.strip()}
    if _WHITELIST_RAW else None
)

# Limite diário por número (msgs recebidas)
DAILY_LIMIT_PER_USER = int(os.getenv("WHATSAPP_DAILY_LIMIT", "30"))

# DB local
DB_PATH = Path(__file__).parent / "whatsapp.db"
HISTORY_MAX_TURNS = 20  # últimos N turnos (user+assistant) por número


# ===========================================================================
# SQLite — histórico por número
# ===========================================================================


def _init_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            phone TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_messages_phone ON messages(phone, created_at);

        CREATE TABLE IF NOT EXISTS daily_usage (
            phone TEXT NOT NULL,
            date_iso TEXT NOT NULL,
            count INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (phone, date_iso)
        );
    """)
    conn.commit()
    conn.close()


_init_db()


def get_history(phone: str) -> list[dict]:
    conn = sqlite3.connect(str(DB_PATH))
    rows = conn.execute(
        "SELECT role, content FROM messages WHERE phone = ? ORDER BY created_at DESC LIMIT ?",
        (phone, HISTORY_MAX_TURNS * 2),
    ).fetchall()
    conn.close()
    # Retornamos em ordem cronológica ascendente
    return [{"role": role, "content": content} for role, content in reversed(rows)]


def append_message(phone: str, role: str, content: str):
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute(
        "INSERT INTO messages (phone, role, content, created_at) VALUES (?, ?, ?, ?)",
        (phone, role, content, time.time()),
    )
    conn.commit()
    conn.close()


def increment_daily(phone: str) -> int:
    """Incrementa contador diário do usuário e retorna o novo valor."""
    from datetime import date
    today = date.today().isoformat()
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute(
        "INSERT INTO daily_usage (phone, date_iso, count) VALUES (?, ?, 1) "
        "ON CONFLICT(phone, date_iso) DO UPDATE SET count = count + 1",
        (phone, today),
    )
    row = conn.execute(
        "SELECT count FROM daily_usage WHERE phone = ? AND date_iso = ?",
        (phone, today),
    ).fetchone()
    conn.commit()
    conn.close()
    return row[0] if row else 0


# ===========================================================================
# Evolution API — envio
# ===========================================================================


async def send_whatsapp_text(phone: str, text: str) -> dict:
    """Envia mensagem de texto via Evolution API."""
    if not EVOLUTION_API_KEY:
        return {"ok": False, "error": "EVOLUTION_API_KEY não configurada"}

    url = f"{EVOLUTION_URL}/message/sendText/{EVOLUTION_INSTANCE}"
    headers = {"apikey": EVOLUTION_API_KEY, "Content-Type": "application/json"}
    payload = {"number": phone, "text": text}

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, headers=headers, timeout=30) as resp:
            body = await resp.text()
            return {"ok": 200 <= resp.status < 300, "status": resp.status, "body": body[:500]}


# ===========================================================================
# Parsing do webhook da Evolution
# ===========================================================================


def parse_incoming(event: dict) -> Optional[dict]:
    """Extrai {phone, text, from_me, push_name} de um webhook da Evolution.

    Retorna None se não for uma mensagem de texto inbound relevante.
    """
    # Evolution envia `event` como "messages.upsert" (ou MESSAGES_UPSERT nas versões novas)
    ev = (event.get("event") or "").lower().replace("_", ".")
    if "messages.upsert" not in ev:
        return None

    data = event.get("data") or {}
    key = data.get("key") or {}
    from_me = bool(key.get("fromMe"))
    if from_me:
        return None  # ignora o que nós mesmos mandamos

    remote_jid = key.get("remoteJid") or ""
    # Formato: "5527999999999@s.whatsapp.net" (pessoa) ou "xxx@g.us" (grupo)
    if remote_jid.endswith("@g.us"):
        return None  # ignora grupos por enquanto
    phone_match = re.match(r"^(\d+)@", remote_jid)
    if not phone_match:
        return None
    phone = phone_match.group(1)

    message = data.get("message") or {}
    text = (
        message.get("conversation")
        or (message.get("extendedTextMessage") or {}).get("text")
        or ""
    ).strip()
    if not text:
        # Não é texto (áudio, imagem, etc) — responderemos fallback
        message_type = data.get("messageType") or ""
        return {
            "phone": phone,
            "text": None,
            "message_type": message_type,
            "push_name": data.get("pushName") or "",
        }

    return {
        "phone": phone,
        "text": text,
        "message_type": data.get("messageType") or "conversation",
        "push_name": data.get("pushName") or "",
    }


def is_allowed(phone: str) -> bool:
    """Checa whitelist. Se WHITELIST is None → aberto."""
    if WHITELIST is None:
        return True
    return phone in WHITELIST


NON_TEXT_REPLY = (
    "👋 Oi! Por enquanto eu só consigo processar mensagens de *texto*. "
    "Manda sua pergunta escrita que eu ajudo — ex:\n\n"
    "_\"Saindo de VIX, oportunidades pro Nordeste no Carnaval 2027?\"_"
)

DAILY_LIMIT_REPLY = (
    "🛑 Você atingiu o limite diário de mensagens comigo. "
    "Amanhã cedo eu volto disponível! 😊"
)

NOT_WHITELISTED_REPLY = (
    "Olá! Esse assistente está em beta fechado. "
    "Se você recebeu esse número por engano, ignore. "
    "Qualquer dúvida, fale com a equipe da Turbinando Suas Milhas."
)
