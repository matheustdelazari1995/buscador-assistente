"""Carrega routes.json + results.json dos 4 serviços (LATAM nac/intl + geral nac/intl)
e oferece funções de busca usadas pelas tools do agente.

Cache em memória com reload automático quando os arquivos mudam no disco
(detecta via mtime). Assim o agente sempre responde com dados frescos.
"""

from __future__ import annotations

import json
import os
import time
from datetime import date as _date
from pathlib import Path
from typing import Optional


# Caminhos na VPS. Em dev local isso é sobrescrito via env DATA_DIRS
DEFAULT_BASES = {
    "latam_nacional":       "/home/gflights/app",
    "latam_internacional":  "/home/gflights/app-internacional",
    "geral_nacional":       "/home/gflights/app-nacional-geral",
    "geral_internacional":  "/home/gflights/app-internacional-geral",
}


_CACHE: dict = {}  # scope_key → {"mtime": float, "routes": {...}, "results": {...}}
_RELOAD_INTERVAL = 60  # segundos — reavalia mtime a cada 1 min
_LAST_CHECK = 0.0


def _bases() -> dict[str, str]:
    override = os.getenv("ASSISTENTE_DATA_DIRS")
    if override:
        # formato: "scope=/path,scope2=/path2"
        out = {}
        for pair in override.split(","):
            k, v = pair.split("=", 1)
            out[k.strip()] = v.strip()
        return out
    return DEFAULT_BASES


def _load_scope(scope_key: str, base: str) -> dict:
    rp = Path(base) / "routes.json"
    sp = Path(base) / "results.json"
    routes = {}
    results = {}
    if rp.exists():
        with open(rp) as f:
            data = json.load(f)
        lst = data.values() if isinstance(data, dict) else data
        routes = {r["id"]: r for r in lst}
    if sp.exists():
        with open(sp) as f:
            results = json.load(f)
    # mtime combinado (mais recente dos 2 arquivos)
    mtime = max(
        rp.stat().st_mtime if rp.exists() else 0,
        sp.stat().st_mtime if sp.exists() else 0,
    )
    return {"mtime": mtime, "routes": routes, "results": results}


def reload_if_stale(force: bool = False) -> dict:
    """Recarrega caches se os arquivos mudaram desde a última checagem."""
    global _LAST_CHECK
    now = time.time()
    if not force and (now - _LAST_CHECK) < _RELOAD_INTERVAL:
        return _CACHE

    for scope, base in _bases().items():
        rp = Path(base) / "routes.json"
        sp = Path(base) / "results.json"
        latest_mtime = max(
            rp.stat().st_mtime if rp.exists() else 0,
            sp.stat().st_mtime if sp.exists() else 0,
        )
        cached = _CACHE.get(scope)
        if force or (cached is None) or latest_mtime > cached["mtime"]:
            try:
                _CACHE[scope] = _load_scope(scope, base)
            except Exception as e:
                print(f"[data_loader] falha ao carregar {scope}: {e}")
                _CACHE.setdefault(scope, {"mtime": 0, "routes": {}, "results": {}})

    _LAST_CHECK = now
    return _CACHE


def _iata_set(xs: Optional[list[str]]) -> Optional[set[str]]:
    if xs is None:
        return None
    return {s.upper() for s in xs}


def search_flights(
    origins: Optional[list[str]] = None,
    dests: Optional[list[str]] = None,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
    max_price_brl: Optional[int] = None,
    scopes: Optional[list[str]] = None,  # subset de ["latam_nacional","latam_internacional","geral_nacional","geral_internacional"]
    limit: int = 20,
) -> list[dict]:
    """Busca voos que baterem com os filtros. Retorna lista de dicts
    ordenada por preço crescente."""
    reload_if_stale()

    orig_set = _iata_set(origins)
    dest_set = _iata_set(dests)
    if scopes:
        scope_set = set(scopes)
    else:
        scope_set = set(_bases().keys())

    d0 = date_start
    d1 = date_end

    results: list[dict] = []

    for scope in scope_set:
        entry = _CACHE.get(scope)
        if not entry:
            continue
        routes = entry["routes"]
        res_all = entry["results"]

        for rid, r in routes.items():
            if orig_set and r["origin"].upper() not in orig_set:
                continue
            if dest_set and r["dest"].upper() not in dest_set:
                continue
            res = res_all.get(rid)
            if not res:
                continue

            # Outbound + inbound (dependendo da direction)
            for leg_key, leg_data in (("outbound", res.get("outbound")), ("inbound", res.get("inbound"))):
                if not leg_data:
                    continue
                # Se inbound, origem/destino invertidos
                if leg_key == "inbound":
                    leg_orig, leg_dest = r["dest"].upper(), r["origin"].upper()
                    # Re-filtro com orientação invertida
                    if orig_set and leg_orig not in orig_set:
                        continue
                    if dest_set and leg_dest not in dest_set:
                        continue
                else:
                    leg_orig, leg_dest = r["origin"].upper(), r["dest"].upper()

                for iso, price in leg_data.items():
                    if d0 and iso < d0:
                        continue
                    if d1 and iso > d1:
                        continue
                    if max_price_brl and price > max_price_brl:
                        continue
                    results.append({
                        "origin": leg_orig,
                        "dest": leg_dest,
                        "date": iso,
                        "price_brl": price,
                        "airline": r.get("airline"),
                        "cabin": r.get("cabin"),
                        "direction": leg_key,
                        "scope": scope,
                        "route_id": rid,
                    })

    # Ordena por preço crescente
    results.sort(key=lambda x: x["price_brl"])
    return results[:limit]


def list_scopes() -> dict[str, dict]:
    """Resumo de quantas rotas/resultados cada scope tem carregado."""
    reload_if_stale()
    out = {}
    for scope, entry in _CACHE.items():
        n_routes = len(entry["routes"])
        n_with_results = sum(1 for rid in entry["routes"] if rid in entry["results"])
        out[scope] = {"routes": n_routes, "routes_com_resultado": n_with_results}
    return out


def find_trip_combinations(
    origins: Optional[list[str]] = None,
    dests: Optional[list[str]] = None,
    outbound_start: Optional[str] = None,
    outbound_end: Optional[str] = None,
    return_start: Optional[str] = None,
    return_end: Optional[str] = None,
    max_total_brl: Optional[int] = None,
    min_stay_days: Optional[int] = None,
    max_stay_days: Optional[int] = None,
    scopes: Optional[list[str]] = None,
    limit: int = 15,
) -> list[dict]:
    """Combina pares ida+volta da mesma rota e retorna ordenados por preço total.

    Uma combinação é um {outbound_date, inbound_date} da mesma rota onde
    inbound_date > outbound_date. Usado pra perguntas tipo 'ir no Carnaval
    e voltar depois do feriado'.
    """
    from datetime import date as _d
    reload_if_stale()

    orig_set = _iata_set(origins)
    dest_set = _iata_set(dests)
    scope_set = set(scopes) if scopes else set(_bases().keys())

    combos: list[dict] = []
    for scope in scope_set:
        entry = _CACHE.get(scope)
        if not entry:
            continue
        for rid, r in entry["routes"].items():
            if orig_set and r["origin"].upper() not in orig_set:
                continue
            if dest_set and r["dest"].upper() not in dest_set:
                continue
            res = entry["results"].get(rid)
            if not res:
                continue
            out_prices = res.get("outbound") or {}
            ret_prices = res.get("inbound") or {}
            if not out_prices or not ret_prices:
                continue

            for out_date, out_price in out_prices.items():
                if outbound_start and out_date < outbound_start:
                    continue
                if outbound_end and out_date > outbound_end:
                    continue
                try:
                    out_d = _d.fromisoformat(out_date)
                except Exception:
                    continue
                for ret_date, ret_price in ret_prices.items():
                    if return_start and ret_date < return_start:
                        continue
                    if return_end and ret_date > return_end:
                        continue
                    try:
                        ret_d = _d.fromisoformat(ret_date)
                    except Exception:
                        continue
                    if ret_d <= out_d:
                        continue  # volta tem que ser depois da ida
                    stay = (ret_d - out_d).days
                    if min_stay_days is not None and stay < min_stay_days:
                        continue
                    if max_stay_days is not None and stay > max_stay_days:
                        continue
                    total = out_price + ret_price
                    if max_total_brl is not None and total > max_total_brl:
                        continue
                    combos.append({
                        "origin": r["origin"].upper(),
                        "dest": r["dest"].upper(),
                        "outbound_date": out_date,
                        "outbound_price_brl": out_price,
                        "inbound_date": ret_date,
                        "inbound_price_brl": ret_price,
                        "total_price_brl": total,
                        "stay_days": stay,
                        "airline": r.get("airline"),
                        "cabin": r.get("cabin"),
                        "scope": scope,
                        "route_id": rid,
                    })

    combos.sort(key=lambda x: x["total_price_brl"])
    return combos[:limit]


def get_route_history(origin: str, dest: str, scope: Optional[str] = None) -> list[dict]:
    """Retorna rotas cadastradas que batem com origin→dest (ou dest→origin)."""
    reload_if_stale()
    o, d = origin.upper(), dest.upper()
    out = []
    scopes = [scope] if scope else list(_bases().keys())
    for s in scopes:
        entry = _CACHE.get(s, {})
        for rid, r in entry.get("routes", {}).items():
            if (r["origin"].upper(), r["dest"].upper()) in {(o, d), (d, o)}:
                res = entry["results"].get(rid, {})
                out.append({
                    "scope": s,
                    "origin": r["origin"],
                    "dest": r["dest"],
                    "airline": r.get("airline"),
                    "direction": r.get("direction"),
                    "cabin": r.get("cabin"),
                    "min_price": r.get("min_price"),
                    "min_by_month": r.get("min_by_month", {}),
                    "last_searched_at": r.get("last_searched_at"),
                    "n_outbound_dates": len(res.get("outbound") or {}),
                    "n_inbound_dates": len(res.get("inbound") or {}),
                })
    return out
