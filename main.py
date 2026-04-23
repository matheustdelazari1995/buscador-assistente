"""Agente conversacional do Buscador Google Flights.

Expõe POST /api/chat que recebe {message, history?, scope?} e retorna
{reply, tool_calls_log?}. Usa Claude Sonnet 4.6 com tool-calling;
as tools batem nas rotas/resultados carregados do disco (data_loader.py).

Segue o padrão de caching do Anthropic SDK (prompt cache no system prompt
+ tools pra reduzir custo em conversas multi-turn).
"""

from __future__ import annotations

import json
import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from anthropic import Anthropic

from data_loader import search_flights, list_scopes, get_route_history, reload_if_stale
from regions import (
    REGIONS, IATA_NAMES, HOLIDAYS, MONTHS_PT,
    region_to_iatas, holiday_date_range, month_date_range,
)


app = FastAPI(title="Buscador Assistente")

# CORS pra frontend de qualquer um dos wrappers poder chamar
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://buscador.turbinandosuasmilhas.com.br",
        "https://buscadorgeral.turbinandosuasmilhas.com.br",
    ],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    print("[WARN] ANTHROPIC_API_KEY não setada — /api/chat vai falhar")

client = Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

MODEL = "claude-sonnet-4-5"  # bom/barato
MAX_TOKENS = 2048
MAX_TOOL_ITERATIONS = 8


# ===========================================================================
# Tools que o LLM pode chamar
# ===========================================================================


TOOLS = [
    {
        "name": "search_flights",
        "description": (
            "Busca voos nos resultados já raspados do Google Flights. Retorna lista ordenada "
            "por preço crescente. Use quando o usuário pede opções concretas de voos "
            "('voos pra FOR em julho', 'mais barato pra Europa', 'Carnaval 2027')."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "origins": {
                    "type": "array", "items": {"type": "string"},
                    "description": "Códigos IATA de origem (ex: ['VIX', 'GRU']). Omita se qualquer origem serve."
                },
                "dests": {
                    "type": "array", "items": {"type": "string"},
                    "description": "Códigos IATA de destino (ex: ['FOR', 'REC', 'SSA']). Omita se qualquer destino serve."
                },
                "date_start": {
                    "type": "string",
                    "description": "Data inicial ISO (YYYY-MM-DD). Inclusivo."
                },
                "date_end": {
                    "type": "string",
                    "description": "Data final ISO (YYYY-MM-DD). Inclusivo."
                },
                "max_price_brl": {
                    "type": "integer",
                    "description": "Preço máximo em reais. Omita se o usuário não limitou."
                },
                "scopes": {
                    "type": "array", "items": {"type": "string",
                        "enum": ["latam_nacional", "latam_internacional",
                                 "geral_nacional", "geral_internacional"]},
                    "description": "Subset de scopes pra buscar. Omita pra buscar em todos."
                },
                "limit": {
                    "type": "integer",
                    "description": "Máximo de resultados a retornar. Default 20, máximo recomendado 50."
                },
            },
        },
    },
    {
        "name": "region_to_iatas",
        "description": (
            "Traduz nome de região (ex: 'Nordeste', 'Europa', 'Caribe') em lista de "
            "códigos IATA. Use antes de search_flights quando o usuário mencionar região."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "region": {"type": "string", "description": "Nome da região."},
            },
            "required": ["region"],
        },
    },
    {
        "name": "holiday_date_range",
        "description": (
            "Retorna intervalo de datas (start, end) pra um feriado brasileiro + ano. "
            "Feriados suportados: carnaval, semana_santa, tiradentes, corpus_christi, "
            "sete_setembro, nossa_senhora, finados, proclamacao_republica, natal, reveillon."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "year": {"type": "integer"},
            },
            "required": ["name", "year"],
        },
    },
    {
        "name": "month_date_range",
        "description": "Intervalo de datas de um mês + ano (ex: 'julho', 2027 → 2027-07-01 a 2027-07-31).",
        "input_schema": {
            "type": "object",
            "properties": {
                "month": {"type": "string", "description": "Nome do mês em português"},
                "year": {"type": "integer"},
            },
            "required": ["month", "year"],
        },
    },
    {
        "name": "list_available_scopes",
        "description": (
            "Lista scopes disponíveis e quantas rotas cada um tem. Use pra entender "
            "o catálogo antes de recomendar algo."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_route_history",
        "description": (
            "Detalhes de uma rota específica (origin→dest). Retorna scopes onde a rota existe, "
            "preço mínimo por mês, última atualização, etc. Útil pra comparar LATAM vs Geral."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "origin": {"type": "string", "description": "IATA origem"},
                "dest": {"type": "string", "description": "IATA destino"},
            },
            "required": ["origin", "dest"],
        },
    },
]


def execute_tool(name: str, args: dict) -> dict:
    """Dispatcher — chama a função Python correspondente à tool."""
    try:
        if name == "search_flights":
            results = search_flights(**args)
            # Enriquece com nome de cidade pra facilitar resposta do LLM
            for r in results:
                r["origin_city"] = IATA_NAMES.get(r["origin"], r["origin"])
                r["dest_city"] = IATA_NAMES.get(r["dest"], r["dest"])
            return {"count": len(results), "results": results}

        if name == "region_to_iatas":
            iatas = region_to_iatas(args["region"])
            if not iatas:
                return {"error": f"Região '{args['region']}' não conhecida. "
                                 f"Disponíveis: {', '.join(sorted(REGIONS.keys()))}"}
            return {"region": args["region"], "iatas": iatas,
                    "cities": [IATA_NAMES.get(i, i) for i in iatas]}

        if name == "holiday_date_range":
            rng = holiday_date_range(args["name"], args["year"])
            if not rng:
                return {"error": f"Feriado '{args['name']}' em {args['year']} não conhecido. "
                                 f"Disponíveis: {', '.join(sorted(HOLIDAYS.keys()))}"}
            return {"start": rng[0], "end": rng[1]}

        if name == "month_date_range":
            rng = month_date_range(args["month"], args["year"])
            if not rng:
                return {"error": f"Mês '{args['month']}' inválido."}
            return {"start": rng[0], "end": rng[1]}

        if name == "list_available_scopes":
            return list_scopes()

        if name == "get_route_history":
            return {"matches": get_route_history(args["origin"], args["dest"])}

        return {"error": f"tool desconhecida: {name}"}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


# ===========================================================================
# System prompt
# ===========================================================================


SYSTEM_PROMPT = """Você é o assistente da **Turbinando Suas Milhas**, uma plataforma brasileira que raspa preços de voos do Google Flights diariamente. Você ajuda a equipe e clientes a encontrar oportunidades de viagem usando os dados já coletados.

**Contexto dos dados:**
- Raspamos 4 catálogos diferentes:
  - `latam_nacional`: voos nacionais operados pela LATAM (~1000 rotas)
  - `latam_internacional`: voos internacionais operados pela LATAM (~850 rotas)
  - `geral_nacional`: voos nacionais de qualquer cia (~1000 rotas)
  - `geral_internacional`: voos internacionais de qualquer cia (~850 rotas)
- Cada rota tem preços por data (calendário do ano todo)
- Preços em **R$ (reais)**, tarifa econômica por padrão
- Para LATAM, podemos estimar milhas: `(preço - 40) / 28.8` em K milhas (mas só mencione se for LATAM)

**Seu papel:**
- Interpretar perguntas em linguagem natural (regiões, datas, orçamentos)
- Chamar as tools apropriadas (region_to_iatas, holiday_date_range, search_flights)
- Apresentar resultados de forma objetiva e útil
- Quando fizer sentido, comparar LATAM vs Geral (ex: "mesma rota tá R$X na LATAM e R$Y na geral")

**Formato da resposta:**
- Use markdown (negrito, listas)
- Liste no máximo 5-10 melhores opções
- Para cada: **rota** · **data** · **preço** · **cia**
- Se for LATAM, mostre também estimativa de milhas
- Sugira próximas perguntas úteis no final

**Se não houver dados:**
- Diga claramente que não temos informação na base
- Sugira rotas próximas/alternativas que existam

**Resuma respostas** — seja direto. Ano atual: 2026. Considere datas futuras (não sugira datas no passado).
"""


# ===========================================================================
# API
# ===========================================================================


class ChatMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []
    scope_hint: Optional[str] = None  # "latam" | "geral" | None (deixa LLM decidir)


class ChatResponse(BaseModel):
    reply: str
    tool_calls: list[dict] = []  # log pra debug


@app.on_event("startup")
async def on_startup():
    # Warm-up dos caches
    reload_if_stale(force=True)
    summary = list_scopes()
    print(f"[startup] dados carregados: {summary}")


@app.get("/api/health")
async def health():
    return {"ok": True, "scopes": list_scopes()}


@app.post("/api/chat")
async def chat(req: ChatRequest) -> ChatResponse:
    if client is None:
        raise HTTPException(500, "ANTHROPIC_API_KEY não configurada")

    # Monta histórico + nova mensagem
    messages = []
    for m in req.history[-20:]:  # limita últimas 20 pra controlar custo
        messages.append({"role": m.role, "content": m.content})

    user_msg = req.message
    if req.scope_hint:
        user_msg = f"[contexto: usuário está olhando o dashboard '{req.scope_hint}']\n\n{user_msg}"
    messages.append({"role": "user", "content": user_msg})

    tool_calls_log: list[dict] = []

    for iteration in range(MAX_TOOL_ITERATIONS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        if response.stop_reason == "end_turn":
            text_parts = [b.text for b in response.content if getattr(b, "type", None) == "text"]
            return ChatResponse(reply="\n".join(text_parts), tool_calls=tool_calls_log)

        if response.stop_reason == "tool_use":
            # Anexa o "assistant" message ao histórico
            messages.append({"role": "assistant", "content": response.content})
            # Executa cada tool_use e monta o user message com tool_results
            tool_results = []
            for block in response.content:
                if getattr(block, "type", None) == "tool_use":
                    result = execute_tool(block.name, block.input)
                    tool_calls_log.append({
                        "name": block.name,
                        "input": block.input,
                        "result_summary": {"keys": list(result.keys())[:5], "count": result.get("count")},
                    })
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result, ensure_ascii=False)[:8000],  # trunca pra evitar context explosion
                    })
            messages.append({"role": "user", "content": tool_results})
            continue

        # stop_reason inesperado
        text_parts = [b.text for b in response.content if getattr(b, "type", None) == "text"]
        return ChatResponse(reply="\n".join(text_parts) or "(sem resposta)", tool_calls=tool_calls_log)

    return ChatResponse(
        reply="(limite de iterações excedido — tente reformular a pergunta)",
        tool_calls=tool_calls_log,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8006)
