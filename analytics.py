"""Analytics — agrega histórico de preços dos 4 services e expõe queries
como 'maiores quedas', 'mais abaixo da média', 'mais baratas' e detalhe
de uma rota específica.

Lê DIRETO do `prices.db` de cada service (modo read-only) e cruza com
metadata de rotas (origem/destino/cia) carregadas via data_loader.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Optional

from data_loader import _bases as _scope_bases, reload_if_stale, _CACHE


# Mapeamento scope → caminho do prices.db
def _db_path(scope: str) -> str:
    base = _scope_bases().get(scope)
    if not base:
        raise ValueError(f"scope desconhecido: {scope}")
    return f"{base}/prices.db"


def _conn(scope: str) -> sqlite3.Connection:
    """Abre conexão SQLite read-only. Cada call abre/fecha — barato."""
    path = _db_path(scope)
    if not Path(path).exists():
        raise FileNotFoundError(f"prices.db não existe pra {scope}: {path}")
    # URI mode com mode=ro pra read-only seguro (não trava o writer do service)
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


# ===========================================================================
# Cache TTL simples
# ===========================================================================


_CACHE_TTL = 300  # 5 min
_RESULT_CACHE: dict[str, tuple[float, object]] = {}


def _cached(key: str, fn, ttl: int = _CACHE_TTL):
    now = time.time()
    if key in _RESULT_CACHE:
        ts, val = _RESULT_CACHE[key]
        if now - ts < ttl:
            return val
    val = fn()
    _RESULT_CACHE[key] = (now, val)
    return val


def invalidate_cache():
    _RESULT_CACHE.clear()


# ===========================================================================
# Helpers — enriquece resultado com metadata da rota (origem/destino/cia)
# ===========================================================================


def _route_meta(scope: str, route_id: str) -> dict:
    """Pega origem/destino/airline/cabin/direction/last_searched_at da rota."""
    reload_if_stale()
    routes = (_CACHE.get(scope) or {}).get("routes", {})
    r = routes.get(route_id)
    if not r:
        return {}
    return {
        "origin": r.get("origin"),
        "dest": r.get("dest"),
        "airline": r.get("airline"),
        "cabin": r.get("cabin"),
        "direction": r.get("direction"),
        "last_searched_at": r.get("last_searched_at"),
    }


def _enrich(scope: str, items: list[dict]) -> list[dict]:
    out = []
    for it in items:
        rid = it["route_id"]
        meta = _route_meta(scope, rid)
        if not meta.get("origin"):
            continue  # rota foi deletada, ignora
        out.append({**it, **meta, "scope": scope})
    return out


# ===========================================================================
# Queries
# ===========================================================================


def top_drops(scope: str, limit: int = 10, min_drop_pct: float = 0.0,
              origin: Optional[str] = None) -> list[dict]:
    """Rotas onde o preço mínimo CAIU mais (em % vs snapshot anterior).

    `min_drop_pct`: filtra resultados com queda menor que esse threshold (negativo).
    `origin`: IATA de origem (ex: 'GRU') pra filtrar só voos saindo dali.
    """
    cache_key = f"drops:{scope}:{limit}:{min_drop_pct}:{origin or ''}"

    def run():
        with _conn(scope) as c:
            cur = c.execute(
                """
                WITH latest AS (
                    SELECT route_id, MAX(searched_at) AS last_at
                    FROM snapshots WHERE min_price IS NOT NULL
                    GROUP BY route_id
                ),
                current AS (
                    SELECT s.route_id, s.searched_at, s.min_price
                    FROM snapshots s
                    JOIN latest l ON s.route_id = l.route_id AND s.searched_at = l.last_at
                    WHERE s.min_price IS NOT NULL
                ),
                ranked AS (
                    SELECT route_id, searched_at, min_price,
                           ROW_NUMBER() OVER (PARTITION BY route_id ORDER BY searched_at DESC) AS rn
                    FROM snapshots WHERE min_price IS NOT NULL
                )
                SELECT c.route_id,
                       c.searched_at AS current_at,
                       c.min_price   AS current_price,
                       p.searched_at AS prev_at,
                       p.min_price   AS prev_price,
                       (c.min_price - p.min_price) AS delta_brl,
                       ROUND(100.0 * (c.min_price - p.min_price) / p.min_price, 1) AS delta_pct
                FROM current c
                JOIN ranked p ON c.route_id = p.route_id AND p.rn = 2
                WHERE p.min_price IS NOT NULL AND p.min_price > 0
                  AND (100.0 * (c.min_price - p.min_price) / p.min_price) <= ?
                ORDER BY delta_pct ASC
                LIMIT ?
                """,
                # Quando filtra por origem, pega mais (até 200) pra não esvaziar
                (min_drop_pct, 200 if origin else limit * 3),
            )
            rows = [dict(r) for r in cur.fetchall()]
        enriched = _enrich(scope, rows)
        if origin:
            enriched = [r for r in enriched if r.get("origin") == origin.upper()]
        return enriched[:limit]

    return _cached(cache_key, run)


def most_below_average(scope: str, limit: int = 10, days: int = 30,
                       min_pct_below: float = 0.0,
                       origin: Optional[str] = None) -> list[dict]:
    """Rotas onde o preço atual está mais abaixo da média histórica dos últimos N dias.

    `min_pct_below`: filtra rotas onde a queda é menor que esse threshold (negativo).
    `origin`: IATA pra filtrar só voos saindo dali.
    """
    cache_key = f"belowavg:{scope}:{limit}:{days}:{min_pct_below}:{origin or ''}"

    def run():
        with _conn(scope) as c:
            cur = c.execute(
                """
                WITH latest AS (
                    SELECT route_id, MAX(searched_at) AS last_at
                    FROM snapshots WHERE min_price IS NOT NULL
                    GROUP BY route_id
                ),
                current AS (
                    SELECT s.route_id, s.searched_at, s.min_price
                    FROM snapshots s
                    JOIN latest l ON s.route_id = l.route_id AND s.searched_at = l.last_at
                    WHERE s.min_price IS NOT NULL
                ),
                hist AS (
                    SELECT route_id, AVG(min_price) AS avg_price,
                           MIN(min_price) AS min_price_hist,
                           MAX(min_price) AS max_price_hist,
                           COUNT(*) AS n
                    FROM snapshots
                    WHERE searched_at >= DATETIME('now', ? || ' days') AND min_price IS NOT NULL
                    GROUP BY route_id
                    HAVING n >= 2
                )
                SELECT c.route_id,
                       c.searched_at AS current_at,
                       c.min_price   AS current_price,
                       ROUND(h.avg_price, 0) AS avg_price,
                       h.min_price_hist,
                       h.max_price_hist,
                       h.n AS sample_count,
                       ROUND(100.0 * (c.min_price - h.avg_price) / h.avg_price, 1) AS pct_vs_avg
                FROM current c JOIN hist h USING (route_id)
                WHERE (100.0 * (c.min_price - h.avg_price) / h.avg_price) <= ?
                ORDER BY pct_vs_avg ASC
                LIMIT ?
                """,
                (f"-{days}", min_pct_below, 200 if origin else limit * 3),
            )
            rows = [dict(r) for r in cur.fetchall()]
        enriched = _enrich(scope, rows)
        if origin:
            enriched = [r for r in enriched if r.get("origin") == origin.upper()]
        return enriched[:limit]

    return _cached(cache_key, run)


def cheapest_now(scope: str, limit: int = 10, origin: Optional[str] = None) -> list[dict]:
    """Top N rotas mais baratas no momento (último snapshot)."""
    cache_key = f"cheapest:{scope}:{limit}:{origin or ''}"

    def run():
        with _conn(scope) as c:
            cur = c.execute(
                """
                WITH latest AS (
                    SELECT route_id, MAX(searched_at) AS last_at
                    FROM snapshots WHERE min_price IS NOT NULL
                    GROUP BY route_id
                )
                SELECT s.route_id, s.searched_at AS current_at, s.min_price AS current_price
                FROM snapshots s
                JOIN latest l ON s.route_id = l.route_id AND s.searched_at = l.last_at
                WHERE s.min_price IS NOT NULL
                ORDER BY s.min_price ASC
                LIMIT ?
                """,
                (200 if origin else limit * 3,),
            )
            rows = [dict(r) for r in cur.fetchall()]
        enriched = _enrich(scope, rows)
        if origin:
            enriched = [r for r in enriched if r.get("origin") == origin.upper()]
        return enriched[:limit]

    return _cached(cache_key, run)


def list_origins(scope: str = "all") -> list[str]:
    """Lista de IATAs de origem únicos com pelo menos 1 snapshot."""
    cache_key = f"origins:{scope}"

    def run():
        scopes = list(_scope_bases().keys()) if scope == "all" else [scope]
        all_origins = set()
        for sc in scopes:
            reload_if_stale()
            routes = (_CACHE.get(sc) or {}).get("routes", {})
            for rid, r in routes.items():
                origin = r.get("origin")
                if origin:
                    all_origins.add(origin.upper())
        return sorted(all_origins)

    return _cached(cache_key, run, ttl=600)


def route_history(scope: str, route_id: str, days: int = 30) -> dict:
    """Histórico de min_price de uma rota nos últimos N dias + estatísticas."""
    cache_key = f"history:{scope}:{route_id}:{days}"

    def run():
        with _conn(scope) as c:
            cur = c.execute(
                """
                SELECT searched_at, min_price
                FROM snapshots
                WHERE route_id = ?
                  AND searched_at >= DATETIME('now', ? || ' days')
                  AND min_price IS NOT NULL
                ORDER BY searched_at ASC
                """,
                (route_id, f"-{days}"),
            )
            points = [{"searched_at": r["searched_at"], "min_price": r["min_price"]}
                      for r in cur.fetchall()]
            # Stats
            prices = [p["min_price"] for p in points]
            stats = {
                "count": len(prices),
                "avg": round(sum(prices) / len(prices), 0) if prices else None,
                "min": min(prices) if prices else None,
                "max": max(prices) if prices else None,
                "current": prices[-1] if prices else None,
                "first": prices[0] if prices else None,
            }
            if stats["current"] is not None and stats["avg"]:
                stats["pct_vs_avg"] = round(100.0 * (stats["current"] - stats["avg"]) / stats["avg"], 1)
        meta = _route_meta(scope, route_id)
        return {"scope": scope, "route_id": route_id, **meta, "points": points, "stats": stats}

    return _cached(cache_key, run, ttl=60)  # detalhe atualiza mais rápido


def all_scopes_top_drops(limit: int = 10, min_drop_pct: float = 0.0,
                          origin: Optional[str] = None) -> list[dict]:
    """Junta top drops dos 4 scopes e retorna os mais expressivos no agregado."""
    all_results = []
    for scope in _scope_bases().keys():
        try:
            all_results.extend(top_drops(scope, limit=limit, min_drop_pct=min_drop_pct, origin=origin))
        except Exception as e:
            print(f"[analytics] erro em top_drops({scope}): {e}")
    all_results.sort(key=lambda x: x.get("delta_pct") or 0)
    return all_results[:limit]


def all_scopes_below_average(limit: int = 10, days: int = 30, min_pct_below: float = 0.0,
                              origin: Optional[str] = None) -> list[dict]:
    all_results = []
    for scope in _scope_bases().keys():
        try:
            all_results.extend(most_below_average(
                scope, limit=limit, days=days, min_pct_below=min_pct_below, origin=origin))
        except Exception as e:
            print(f"[analytics] erro em below_avg({scope}): {e}")
    all_results.sort(key=lambda x: x.get("pct_vs_avg") or 0)
    return all_results[:limit]


def all_scopes_cheapest(limit: int = 10, origin: Optional[str] = None) -> list[dict]:
    all_results = []
    for scope in _scope_bases().keys():
        try:
            all_results.extend(cheapest_now(scope, limit=limit, origin=origin))
        except Exception as e:
            print(f"[analytics] erro em cheapest({scope}): {e}")
    all_results.sort(key=lambda x: x.get("current_price") or 99999999)
    return all_results[:limit]
