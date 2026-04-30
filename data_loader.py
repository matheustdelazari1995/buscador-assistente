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
    scopes: Optional[list[str]] = None,
    limit: int = 20,
) -> list[dict]:
    """Retorna entradas de preço ordenadas por preço crescente.

    IMPORTANTE — cada resultado tem `pricing_type`:

    - `round_trip_bundled` (internacional): o preço JÁ é da viagem
      completa ida+volta. `date` é a ida; `inbound_date` é a volta
      calculada (ida + `stay_days`). `price_brl` é o total.
    - `one_way` (nacional): preço é de 1 trecho. Não confunde com total —
      se quiser ida+volta, use `find_trip_combinations`.
    """
    from datetime import date as _d, timedelta as _td
    reload_if_stale()

    orig_set = _iata_set(origins)
    dest_set = _iata_set(dests)
    scope_set = set(scopes) if scopes else set(_bases().keys())
    d0 = date_start
    d1 = date_end

    results: list[dict] = []

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

            # Formato round-trip bundled (internacional)?
            by_dur_raw = res.get("outbound_by_duration") or {}
            by_dur = {}
            for k, v in by_dur_raw.items():
                try:
                    dur = int(k)
                except (ValueError, TypeError):
                    continue
                if v:
                    by_dur[dur] = v

            if by_dur:
                # Round-trip bundled: cada data → melhor duração (menor preço)
                best_per_iso: dict[str, tuple[int, int]] = {}  # iso → (dur, price)
                for dur, prices in by_dur.items():
                    for iso, p in prices.items():
                        cur = best_per_iso.get(iso)
                        if cur is None or p < cur[1]:
                            best_per_iso[iso] = (dur, p)

                for iso, (dur, price) in best_per_iso.items():
                    if d0 and iso < d0:
                        continue
                    if d1 and iso > d1:
                        continue
                    if max_price_brl and price > max_price_brl:
                        continue
                    try:
                        out_d = _d.fromisoformat(iso)
                        ret_iso = (out_d + _td(days=dur)).isoformat()
                    except Exception:
                        continue
                    results.append({
                        "origin": r["origin"].upper(),
                        "dest": r["dest"].upper(),
                        "date": iso,               # data de ida
                        "inbound_date": ret_iso,    # data de volta calculada
                        "stay_days": dur,
                        "price_brl": price,         # total ida+volta
                        "airline": r.get("airline"),
                        "cabin": r.get("cabin"),
                        "scope": scope,
                        "route_id": rid,
                        "pricing_type": "round_trip_bundled",
                    })
            else:
                # One-way legs (nacional): retorna cada trecho como entrada separada
                for leg_key, leg_data in (
                    ("outbound", res.get("outbound")),
                    ("inbound", res.get("inbound")),
                ):
                    if not leg_data:
                        continue
                    if leg_key == "inbound":
                        leg_orig, leg_dest = r["dest"].upper(), r["origin"].upper()
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
                            "pricing_type": "one_way",
                        })

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
    """Combina pares ida+volta e retorna ordenados por preço total.

    Trata 2 estruturas de preço distintas:

    - **Internacional (round-trip bundled):** quando existe `outbound_by_duration`
      nos resultados, cada entrada `{duration: {date: price}}` é JÁ o preço total
      da viagem ida+volta (LATAM internacional busca roundtrip direto). Aqui a
      data de volta é calculada como `outbound_date + duration dias`.
    - **Nacional (one-way legs):** cada data de `outbound` é o preço de 1 trecho;
      a volta é buscada separadamente invertendo origem/destino e o preço total
      é a SOMA dos 2 trechos.
    """
    from datetime import date as _d, timedelta as _td
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

            # Detecta qual formato de preço a rota tem
            by_dur_raw = res.get("outbound_by_duration") or {}
            by_dur: dict[int, dict[str, int]] = {}
            for k, v in by_dur_raw.items():
                try:
                    dur = int(k)
                except (ValueError, TypeError):
                    continue
                if v:
                    by_dur[dur] = v

            if by_dur:
                # ========== Round-trip bundled (internacional) ==========
                # Preço JÁ é total ida+volta. Data de volta = ida + duração.
                for dur, prices in by_dur.items():
                    if min_stay_days is not None and dur < min_stay_days:
                        continue
                    if max_stay_days is not None and dur > max_stay_days:
                        continue
                    for out_iso, total_price in prices.items():
                        if outbound_start and out_iso < outbound_start:
                            continue
                        if outbound_end and out_iso > outbound_end:
                            continue
                        try:
                            out_d = _d.fromisoformat(out_iso)
                        except Exception:
                            continue
                        ret_d = out_d + _td(days=dur)
                        ret_iso = ret_d.isoformat()
                        if return_start and ret_iso < return_start:
                            continue
                        if return_end and ret_iso > return_end:
                            continue
                        if max_total_brl is not None and total_price > max_total_brl:
                            continue
                        combos.append({
                            "origin": r["origin"].upper(),
                            "dest": r["dest"].upper(),
                            "outbound_date": out_iso,
                            "inbound_date": ret_iso,
                            "total_price_brl": total_price,
                            "stay_days": dur,
                            "airline": r.get("airline"),
                            "cabin": r.get("cabin"),
                            "scope": scope,
                            "route_id": rid,
                            "pricing_type": "round_trip_bundled",
                            # Não expomos outbound/inbound_price separado —
                            # o scrape do internacional não dá isso.
                        })
            else:
                # ========== One-way legs (nacional) ==========
                out_prices = res.get("outbound") or {}
                ret_prices = res.get("inbound") or {}
                if not out_prices or not ret_prices:
                    continue
                for out_iso, out_price in out_prices.items():
                    if outbound_start and out_iso < outbound_start:
                        continue
                    if outbound_end and out_iso > outbound_end:
                        continue
                    try:
                        out_d = _d.fromisoformat(out_iso)
                    except Exception:
                        continue
                    for ret_iso, ret_price in ret_prices.items():
                        if return_start and ret_iso < return_start:
                            continue
                        if return_end and ret_iso > return_end:
                            continue
                        try:
                            ret_d = _d.fromisoformat(ret_iso)
                        except Exception:
                            continue
                        if ret_d <= out_d:
                            continue
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
                            "outbound_date": out_iso,
                            "outbound_price_brl": out_price,
                            "inbound_date": ret_iso,
                            "inbound_price_brl": ret_price,
                            "total_price_brl": total,
                            "stay_days": stay,
                            "airline": r.get("airline"),
                            "cabin": r.get("cabin"),
                            "scope": scope,
                            "route_id": rid,
                            "pricing_type": "two_one_ways",
                        })

    combos.sort(key=lambda x: x["total_price_brl"])
    return combos[:limit]


def quiz_search(
    scopes: Optional[list[str]] = None,        # subset de scopes (default: todos)
    origins: Optional[list[str]] = None,       # ['GIG'] ou None pra qualquer
    dests: Optional[list[str]] = None,         # ['FOR','REC',...] ou None
    months: Optional[list[int]] = None,        # ex: [7, 8] pra julho+agosto
    year: Optional[int] = None,                # opcional: ano âncora (default = ano atual).
                                                # Quando month < mês atual, usa year+1.
    limit: int = 10,
) -> list[dict]:
    """Busca AGREGADA POR ROTA pra usar no funil de quiz.

    `months` aceita lista (multi-mês) — datas são filtradas pra estarem
    em PELO MENOS UM dos meses passados. O ano de cada mês é derivado
    automaticamente: meses >= mês atual ficam no ano corrente; meses
    < mês atual avançam pra ano+1 (rolling 12 meses pra frente).
    Sem months → ano inteiro do `year` (ou ano atual).
    """
    from datetime import date as _d, timedelta as _td, datetime as _dt
    reload_if_stale()

    orig_set = _iata_set(origins)
    dest_set = _iata_set(dests)
    scope_set = set(scopes) if scopes else set(_bases().keys())

    # Computa ranges de datas: lista de (start_iso, end_iso) — 1 por mês.
    # Cada mês tem ano derivado: se m >= mês_atual → year, senão → year+1.
    ranges: list[tuple[str, str]] = []
    if months:
        today = _dt.now().date()
        anchor_year = year if year else today.year
        cur_month = today.month
        for m in months:
            y = anchor_year if m >= cur_month else anchor_year + 1
            ds = _d(y, m, 1)
            de = _d(y + (1 if m == 12 else 0),
                    1 if m == 12 else m + 1, 1) - _td(days=1)
            ranges.append((ds.isoformat(), de.isoformat()))
    elif year:
        ranges.append((f"{year}-01-01", f"{year}-12-31"))
    # se sem year e sem months: ranges vazio = sem filtro (todas as datas)

    def _in_any_range(date_iso: str) -> bool:
        if not ranges:
            return True
        for ds, de in ranges:
            if ds <= date_iso <= de:
                return True
        return False

    by_route: dict[tuple, dict] = {}  # (scope, origin, dest, airline) -> agg

    for scope in scope_set:
        entry = _CACHE.get(scope)
        if not entry:
            continue
        for rid, r in entry["routes"].items():
            origin = (r.get("origin") or "").upper()
            dest = (r.get("dest") or "").upper()
            if orig_set and origin not in orig_set:
                continue
            if dest_set and dest not in dest_set:
                continue
            res = entry["results"].get(rid)
            if not res:
                continue

            out = res.get("outbound") or {}
            inb = res.get("inbound") or {}
            by_dur = res.get("outbound_by_duration") or {}

            out_in = [(d, p) for d, p in out.items() if _in_any_range(d)]
            inb_in = [(d, p) for d, p in inb.items() if _in_any_range(d)]

            # Computa min_price de acordo com tipo de rota:
            # - intl bundled (`outbound_by_duration` existe): outbound já é total ida+volta
            # - nacional roundtrip (out + inb separados): SOMA cheapest ida + cheapest volta
            # - one-way: usa só o trecho disponível
            min_price = None
            cheapest_date = None
            cheapest_return_date = None
            n_dates = 0
            pricing_type = None
            top_combos: list[dict] = []

            if by_dur:
                # Internacional bundled — outbound já é total ida+volta. Cada (dur, date)
                # é um par real: data de volta = data + dur dias.
                if not out_in:
                    continue
                cheapest_out = min(out_in, key=lambda x: x[1])
                min_price = cheapest_out[1]
                cheapest_date = cheapest_out[0]
                n_dates = len(out_in)
                pricing_type = "round_trip_bundled"

                # Monta lista de combos reais (out_iso, ret_iso, total) com filtro de range
                # nas duas pontas. Pega top 3 mais baratos com (out, ret) únicos.
                combo_list: list[dict] = []
                for k, prices in by_dur.items():
                    try:
                        dur = int(k)
                    except (ValueError, TypeError):
                        continue
                    if not prices:
                        continue
                    for out_iso, total in prices.items():
                        if not _in_any_range(out_iso):
                            continue
                        try:
                            ret_iso = (_d.fromisoformat(out_iso) + _td(days=dur)).isoformat()
                        except Exception:
                            continue
                        # Volta também precisa estar no range selecionado
                        if not _in_any_range(ret_iso):
                            continue
                        combo_list.append({
                            "outbound_date": out_iso,
                            "inbound_date": ret_iso,
                            "stay_days": dur,
                            "total_price": total,
                        })
                combo_list.sort(key=lambda x: x["total_price"])
                seen: set = set()
                for c in combo_list:
                    key2 = (c["outbound_date"], c["inbound_date"])
                    if key2 in seen:
                        continue
                    seen.add(key2)
                    top_combos.append(c)
                    if len(top_combos) >= 3:
                        break
                # Atualiza cheapest_date / cheapest_return_date com base no top combo
                # (datas reais pareadas — out da combinação mais barata)
                if top_combos:
                    cheapest_date = top_combos[0]["outbound_date"]
                    cheapest_return_date = top_combos[0]["inbound_date"]
                    min_price = top_combos[0]["total_price"]
            elif out and inb:
                # Nacional roundtrip — pares (out, ret) com ret > out e duração 2-30 dias.
                if not out_in:
                    continue  # se não tem ida no range, ignora
                # Gera pares filtrando ambas as pontas pelo range selecionado
                combo_list: list[dict] = []
                # inb_in já está filtrado; se vazio, fallback pra inb completo (ex: filtro mês curtou volta)
                inb_pool = inb_in if inb_in else list(inb.items())
                for out_iso, out_price in out_in:
                    try:
                        out_d = _d.fromisoformat(out_iso)
                    except Exception:
                        continue
                    for ret_iso, ret_price in inb_pool:
                        try:
                            ret_d = _d.fromisoformat(ret_iso)
                        except Exception:
                            continue
                        stay = (ret_d - out_d).days
                        if stay < 2 or stay > 30:
                            continue
                        combo_list.append({
                            "outbound_date": out_iso,
                            "outbound_price": out_price,
                            "inbound_date": ret_iso,
                            "inbound_price": ret_price,
                            "stay_days": stay,
                            "total_price": out_price + ret_price,
                        })
                combo_list.sort(key=lambda x: x["total_price"])
                seen: set = set()
                for c in combo_list:
                    key2 = (c["outbound_date"], c["inbound_date"])
                    if key2 in seen:
                        continue
                    seen.add(key2)
                    top_combos.append(c)
                    if len(top_combos) >= 3:
                        break

                if top_combos:
                    min_price = top_combos[0]["total_price"]
                    cheapest_date = top_combos[0]["outbound_date"]
                    cheapest_return_date = top_combos[0]["inbound_date"]
                else:
                    # Fallback: sem pares válidos (duração fora 2-30) → soma min/min como antes
                    cheapest_out = min(out_in, key=lambda x: x[1])
                    cheapest_inb_tuple = min(inb_pool, key=lambda x: x[1])
                    min_price = cheapest_out[1] + cheapest_inb_tuple[1]
                    cheapest_date = cheapest_out[0]
                    cheapest_return_date = cheapest_inb_tuple[0]
                n_dates = len(out_in)
                pricing_type = "round_trip_sum"
            elif out_in:
                # Só ida
                cheapest_out = min(out_in, key=lambda x: x[1])
                min_price = cheapest_out[1]
                cheapest_date = cheapest_out[0]
                n_dates = len(out_in)
                pricing_type = "one_way"
            elif inb_in:
                # Só volta
                cheapest_inb = min(inb_in, key=lambda x: x[1])
                min_price = cheapest_inb[1]
                cheapest_date = cheapest_inb[0]
                n_dates = len(inb_in)
                pricing_type = "one_way"
            else:
                continue

            key = (scope, origin, dest, r.get("airline"))
            existing = by_route.get(key)
            if existing is None or min_price < existing["min_price"]:
                by_route[key] = {
                    "scope": scope,
                    "route_id": rid,
                    "origin": origin,
                    "dest": dest,
                    "airline": r.get("airline"),
                    "cabin": r.get("cabin"),
                    "direction": r.get("direction"),
                    "min_price": min_price,
                    "cheapest_date": cheapest_date,
                    "cheapest_return_date": cheapest_return_date,
                    "n_dates": n_dates,
                    "pricing_type": pricing_type,
                    "top_combos": top_combos,
                    "last_searched_at": r.get("last_searched_at"),
                }

    # Enriquece com nome de cidade pra UI
    from regions import IATA_NAMES
    items = list(by_route.values())
    for it in items:
        it["origin_city"] = IATA_NAMES.get(it["origin"], it["origin"])
        it["dest_city"] = IATA_NAMES.get(it["dest"], it["dest"])
    items.sort(key=lambda x: x["min_price"])
    return items[:limit]


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
