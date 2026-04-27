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

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from anthropic import Anthropic

import whatsapp as wa
import analytics

from data_loader import (
    search_flights, find_trip_combinations,
    list_scopes, get_route_history, reload_if_stale,
)
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

MODEL = "claude-haiku-4-5"  # ~3-5x mais barato que Sonnet, ótimo pra tool-calling estruturado
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
        "name": "find_trip_combinations",
        "description": (
            "Combina ida + volta da MESMA rota e retorna pacotes ordenados por preço total. "
            "USE ESTA (não search_flights) quando o usuário pede uma viagem com retorno "
            "(qualquer menção a feriado, fim de semana, período específico, 'ida e volta', "
            "'quero passar X dias em Y'). Retorna {outbound_date, inbound_date, total_price_brl, stay_days}."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "origins": {"type": "array", "items": {"type": "string"},
                    "description": "IATAs de origem"},
                "dests": {"type": "array", "items": {"type": "string"},
                    "description": "IATAs de destino"},
                "outbound_start": {"type": "string", "description": "Início do range de ida (ISO YYYY-MM-DD)"},
                "outbound_end": {"type": "string", "description": "Fim do range de ida (ISO)"},
                "return_start": {"type": "string", "description": "Início do range de volta"},
                "return_end": {"type": "string", "description": "Fim do range de volta"},
                "min_stay_days": {"type": "integer",
                    "description": "Mínimo de dias entre ida e volta"},
                "max_stay_days": {"type": "integer",
                    "description": "Máximo de dias entre ida e volta"},
                "max_total_brl": {"type": "integer",
                    "description": "Preço total máximo (ida + volta) em reais"},
                "scopes": {"type": "array", "items": {"type": "string"},
                    "description": "Subset de scopes. Omita pra buscar em todos."},
                "limit": {"type": "integer", "description": "Máx resultados. Default 15."},
            },
        },
    },
    {
        "name": "list_holidays",
        "description": (
            "Lista feriados brasileiros cadastrados com suas datas. Use quando o usuário "
            "perguntar sobre 'algum feriado' sem especificar qual, ou pedir recomendação ampla."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "year": {"type": "integer",
                    "description": "Filtra por ano específico. Omita pra listar todos."},
                "from_date": {"type": "string",
                    "description": "Retorna apenas feriados a partir desta data (YYYY-MM-DD). Útil pra não sugerir feriados passados."},
            },
        },
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

        if name == "find_trip_combinations":
            combos = find_trip_combinations(**args)
            for c in combos:
                c["origin_city"] = IATA_NAMES.get(c["origin"], c["origin"])
                c["dest_city"] = IATA_NAMES.get(c["dest"], c["dest"])
            return {"count": len(combos), "combinations": combos}

        if name == "list_holidays":
            out = []
            for h_name, years in HOLIDAYS.items():
                for yr, (start, end) in years.items():
                    if args.get("year") and yr != args["year"]:
                        continue
                    if args.get("from_date") and end < args["from_date"]:
                        continue
                    out.append({"name": h_name, "year": yr, "start": start, "end": end})
            out.sort(key=lambda x: x["start"])
            return {"holidays": out}

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
- Chamar as tools apropriadas e apresentar resultados de forma objetiva

**🔴 REGRA DE OURO — quando o usuário pergunta sobre "passagem", "viagem", "ir pra X", "voar pra Y" (SEM especificar que é só ida):**

👉 **SEMPRE use `find_trip_combinations`, NUNCA `search_flights`.**

Pessoas planejam VIAGENS (ida+volta), não pernas soltas. Mesmo perguntas vagas como _"qual a melhor passagem pra Europa em julho"_ devem ser tratadas como ida+volta e retornadas como pacote.

Só use `search_flights` quando o usuário EXPLICITAMENTE disser:
- "só ida", "só trecho", "one-way"
- "só pra ir", "passagem só de ida"
- "só a volta", "só trecho de volta"

**Outras regras:**
- Se perguntar sobre "algum feriado" sem especificar: primeiro `list_holidays(year=X, from_date=hoje)`, depois `find_trip_combinations` pra cada feriado relevante
- Para regiões ("Nordeste", "Europa"): `region_to_iatas` primeiro, depois use o array em `dests`

**⚠️ IMPORTANTE — semântica de preço na resposta:**

Os preços que as tools retornam (`total_price_brl`) já são **SEMPRE TOTAL DA VIAGEM IDA+VOLTA**, mesmo quando a fonte interna é diferente:

- **Scopes internacionais** (`latam_internacional`, `geral_internacional`): scraper busca round-trip direto. `total_price_brl` é o preço completo. `pricing_type: "round_trip_bundled"`.
- **Scopes nacionais** (`latam_nacional`, `geral_nacional`): scraper busca 2 trechos. `total_price_brl` já vem SOMADO. `pricing_type: "two_one_ways"`.

**NUNCA diga "(ida)" ou "one-way" num resultado de find_trip_combinations — ele SEMPRE representa uma viagem completa. Apresente sempre com ida em `DD/MM`, volta em `DD/MM` e preço total.**

**Ao usar find_trip_combinations pra feriado:**
- outbound_start/end: do primeiro dia do feriado OU 1 dia antes → até o primeiro dia
- return_start/end: do último dia do feriado → até 2-3 dias depois
- Exemplo Carnaval 2027 (6-10/fev): outbound_start=2027-02-05, outbound_end=2027-02-07, return_start=2027-02-10, return_end=2027-02-13

**Formato da resposta:**
- Use markdown (negrito, listas)
- Liste 5-8 melhores opções (pacotes completos ida+volta, não só trechos soltos)
- Para cada pacote: **origem → destino** · ida em `DD/MM` · volta em `DD/MM` (X dias) · **R$ total** · cia
- Destaque o **campeão** (mais barato) e adicione alternativas relevantes
- Sugira próximas perguntas úteis no final

**Estimativa de milhas (SÓ para rotas LATAM):**
- Quando `airline = "LATAM"` num resultado, calcule milhas aproximadas com a fórmula: `milhas_k ≈ (preço_total_brl - 40) / 28.8` (resultado em milhares, arredonde pra cima)
- Mostre uma **linha única consolidada** no final da resposta (não em cada item), tipo:
  _"💳 As opções LATAM ficam na faixa de **XX–YY mil milhas** (ida+volta). Vale conferir direto no site da LatamPass pra valor exato."_
- Se tiver só uma opção LATAM, use: _"💳 Provavelmente na faixa de **XX mil milhas** (ida+volta) — vale conferir direto no site da LatamPass."_
- Para resultados com `airline = null` (scopes `*_geral`), **nunca mencione milhas** — essas cias não permitem resgate em LATAM Pass

**Se não houver dados:**
- Diga claramente e sugira alternativas (destinos próximos, outras datas)

**Resuma** — seja direto. Ano atual: **2026**. Nunca sugira datas passadas. Hoje é depois de abril/2026.
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


WHATSAPP_PROMPT_SUFFIX = """

**Formato especial WhatsApp (use SEMPRE quando receber instrução de WhatsApp):**
- NUNCA use markdown de títulos (`#`, `##`) — WhatsApp não renderiza
- Use `*texto*` (um asterisco) pra negrito, NÃO `**texto**`
- Use `_texto_` pra itálico
- Listas usam `•` ou `-`
- Sem separadores `---`
- Pode usar emojis moderadamente 😊
- Mostre 3-5 melhores opções (não encurte a ponto de omitir info essencial)

**Nunca cortes dados importantes pra economizar linhas.** Cada opção de viagem precisa ter: origem→destino, data de ida, data de volta, duração (dias) e preço total. Se for LATAM, inclua a faixa de milhas no fim.
"""


async def run_chat(message: str, history: list[dict], scope_hint: Optional[str] = None,
                    format_hint: str = "markdown") -> tuple[str, list[dict]]:
    """Executa o loop de tool-calling e retorna (reply, tool_calls_log).

    format_hint: "markdown" (web) ou "whatsapp" (curto + formato WA).
    """
    if client is None:
        raise RuntimeError("ANTHROPIC_API_KEY não configurada")

    system = SYSTEM_PROMPT
    if format_hint == "whatsapp":
        system = SYSTEM_PROMPT + WHATSAPP_PROMPT_SUFFIX

    messages = []
    for m in history[-20:]:
        messages.append({"role": m["role"] if isinstance(m, dict) else m.role,
                         "content": m["content"] if isinstance(m, dict) else m.content})

    user_msg = message
    if scope_hint:
        user_msg = f"[contexto: usuário está olhando o dashboard '{scope_hint}']\n\n{user_msg}"
    if format_hint == "whatsapp":
        user_msg = f"[canal: WhatsApp — responda curto e com formato WA]\n\n{user_msg}"
    messages.append({"role": "user", "content": user_msg})

    tool_calls_log: list[dict] = []

    for iteration in range(MAX_TOOL_ITERATIONS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=system,
            tools=TOOLS,
            messages=messages,
        )

        if response.stop_reason == "end_turn":
            text_parts = [b.text for b in response.content if getattr(b, "type", None) == "text"]
            return "\n".join(text_parts), tool_calls_log

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
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
                        "content": json.dumps(result, ensure_ascii=False)[:8000],
                    })
            messages.append({"role": "user", "content": tool_results})
            continue

        text_parts = [b.text for b in response.content if getattr(b, "type", None) == "text"]
        return "\n".join(text_parts) or "(sem resposta)", tool_calls_log

    return "(limite de iterações excedido — tente reformular a pergunta)", tool_calls_log


@app.post("/api/chat")
async def chat(req: ChatRequest) -> ChatResponse:
    try:
        history = [{"role": m.role, "content": m.content} for m in req.history]
        reply, log = await run_chat(req.message, history, req.scope_hint, format_hint="markdown")
        return ChatResponse(reply=reply, tool_calls=log)
    except RuntimeError as e:
        raise HTTPException(500, str(e))


# ===========================================================================
# WhatsApp — webhook da Evolution API
# ===========================================================================


async def _handle_whatsapp_message(phone: str, text: Optional[str], message_type: str):
    """Executado em background após retornar 200 do webhook."""
    try:
        # 1) Não-texto → fallback curto
        if text is None:
            await wa.send_whatsapp_text(phone, wa.NON_TEXT_REPLY)
            return

        # 2) Whitelist
        if not wa.is_allowed(phone):
            await wa.send_whatsapp_text(phone, wa.NOT_WHITELISTED_REPLY)
            return

        # 3) Daily limit
        count = wa.increment_daily(phone)
        if count > wa.DAILY_LIMIT_PER_USER:
            await wa.send_whatsapp_text(phone, wa.DAILY_LIMIT_REPLY)
            return

        # 4) Persiste a msg do usuário + puxa histórico
        wa.append_message(phone, "user", text)
        history = wa.get_history(phone)[:-1]  # histórico ANTES dessa msg (já foi salva agora)

        # 5) Chama o agente com formato WhatsApp
        reply, _log = await run_chat(
            message=text,
            history=history,
            scope_hint=None,
            format_hint="whatsapp",
        )

        # 6) Persiste + envia
        wa.append_message(phone, "assistant", reply)
        await wa.send_whatsapp_text(phone, reply)
    except Exception as e:
        print(f"[whatsapp] erro processando msg de {phone}: {type(e).__name__}: {e}")
        # Tenta avisar o usuário
        try:
            await wa.send_whatsapp_text(
                phone, "⚠️ Ops, tive um problema técnico. Tenta de novo em alguns instantes.")
        except Exception:
            pass


@app.post("/api/whatsapp/webhook")
async def whatsapp_webhook(request: Request, background: BackgroundTasks):
    """Recebe eventos da Evolution API (WhatsApp). Valida o secret via
    query param `?secret=...` e processa em background pra responder 200 rápido."""
    secret = request.query_params.get("secret", "")
    if secret != wa.WEBHOOK_SECRET:
        raise HTTPException(401, "invalid secret")

    try:
        event = await request.json()
    except Exception:
        raise HTTPException(400, "invalid json")

    parsed = wa.parse_incoming(event)
    if parsed is None:
        return {"ok": True, "skipped": True}

    # Dispara em background pra Evolution não dar timeout
    background.add_task(
        _handle_whatsapp_message,
        parsed["phone"], parsed["text"], parsed["message_type"],
    )
    return {"ok": True, "queued": True}


# ===========================================================================
# Analytics — dashboard de histórico de preços
# ===========================================================================


@app.get("/api/analytics/top-drops")
async def api_top_drops(scope: str = "all", limit: int = 10, min_drop_pct: float = 0.0,
                         origin: Optional[str] = None):
    """Top N rotas com maior queda de preço vs snapshot anterior.

    `scope`: 'all' = agrega 4 scopes; ou um scope específico.
    `min_drop_pct`: filtra resultados com queda <= esse %. Ex: -10 = só rotas que caíram >=10%.
    `origin`: IATA pra filtrar voos saindo de uma cidade (ex: 'GRU').
    """
    try:
        if scope == "all":
            return {"items": analytics.all_scopes_top_drops(limit=limit, min_drop_pct=min_drop_pct, origin=origin)}
        return {"items": analytics.top_drops(scope, limit=limit, min_drop_pct=min_drop_pct, origin=origin)}
    except Exception as e:
        raise HTTPException(500, f"{type(e).__name__}: {e}")


@app.get("/api/analytics/below-average")
async def api_below_avg(scope: str = "all", limit: int = 10, days: int = 30,
                         min_pct_below: float = 0.0, origin: Optional[str] = None):
    """Rotas com preço atual mais abaixo da média histórica."""
    try:
        if scope == "all":
            return {"items": analytics.all_scopes_below_average(
                limit=limit, days=days, min_pct_below=min_pct_below, origin=origin)}
        return {"items": analytics.most_below_average(
            scope, limit=limit, days=days, min_pct_below=min_pct_below, origin=origin)}
    except Exception as e:
        raise HTTPException(500, f"{type(e).__name__}: {e}")


@app.get("/api/analytics/cheapest")
async def api_cheapest(scope: str = "all", limit: int = 10, origin: Optional[str] = None):
    """Top N rotas mais baratas no momento."""
    try:
        if scope == "all":
            return {"items": analytics.all_scopes_cheapest(limit=limit, origin=origin)}
        return {"items": analytics.cheapest_now(scope, limit=limit, origin=origin)}
    except Exception as e:
        raise HTTPException(500, f"{type(e).__name__}: {e}")


@app.get("/api/analytics/origins")
async def api_origins(scope: str = "all"):
    """Lista de cidades de origem (IATAs) disponíveis pra filtrar."""
    try:
        return {"origins": analytics.list_origins(scope)}
    except Exception as e:
        raise HTTPException(500, f"{type(e).__name__}: {e}")


@app.get("/api/analytics/route/{scope}/{route_id}/history")
async def api_route_history(scope: str, route_id: str, days: int = 30):
    """Histórico detalhado de uma rota: pontos de preço + estatísticas."""
    try:
        return analytics.route_history(scope, route_id, days=days)
    except Exception as e:
        raise HTTPException(500, f"{type(e).__name__}: {e}")


@app.post("/api/analytics/cache/invalidate")
async def api_invalidate_cache():
    """Limpa o cache de analytics (útil pra testes)."""
    analytics.invalidate_cache()
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8006)
