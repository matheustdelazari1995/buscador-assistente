"""Mapeamentos de regiões geográficas → códigos IATA e calendário de feriados BR.

Usado pelo agente pra traduzir perguntas em linguagem natural ("Nordeste",
"Carnaval 2027") em filtros concretos que batem com as rotas cadastradas.
"""

from datetime import date, timedelta
from typing import Optional


# Conjuntos pra classificar regiões em UI (filtros do quiz por scope)
BR_REGIONS = {"nordeste", "sul", "sudeste", "centro_oeste", "norte"}
INTL_REGIONS_HINT = None  # qualquer região não-BR é internacional


REGIONS = {
    # Brasil
    "nordeste": ["FOR", "REC", "SSA", "NAT", "MCZ", "JPA", "AJU", "SLZ", "THE"],
    "sul": ["POA", "FLN", "CWB", "NVT", "JOI"],
    "sudeste": ["GRU", "CGH", "VCP", "GIG", "SDU", "CNF", "PLU", "VIX", "UDI"],
    "centro_oeste": ["BSB", "CGB", "CGR", "GYN"],
    "norte": ["BEL", "MAO", "MCP", "MAB", "PVH", "BVB", "STM", "RBR", "PMW"],

    # Internacional
    "europa": ["LIS", "OPO", "MAD", "BCN", "CDG", "ORY", "FCO", "MXP",
               "LHR", "LGW", "AMS", "FRA", "MUC", "ZRH", "VIE", "DUB"],
    "portugal": ["LIS", "OPO"],
    "espanha": ["MAD", "BCN"],
    "franca": ["CDG", "ORY"],
    "italia": ["FCO", "MXP"],
    "reino_unido": ["LHR", "LGW", "MAN"],
    "eua": ["MIA", "JFK", "EWR", "LGA", "LAX", "SFO", "ORD", "BOS", "ATL",
            "DFW", "IAH", "IAD", "DCA", "SEA", "LAS", "MCO", "FLL"],
    "estados_unidos": ["MIA", "JFK", "EWR", "LGA", "LAX", "SFO", "ORD", "BOS",
                       "ATL", "DFW", "IAH", "IAD", "DCA", "SEA", "LAS", "MCO", "FLL"],
    "caribe": ["CCS", "MBJ", "HAV", "SJU", "SDQ", "PUJ", "AUA", "NAS", "CUN"],
    "america_sul": ["SCL", "EZE", "AEP", "MVD", "BOG", "LIM", "UIO", "GYE", "LPB", "ASU"],
    "argentina": ["EZE", "AEP"],
    "chile": ["SCL"],
    "uruguai": ["MVD"],
    "peru": ["LIM"],
    "colombia": ["BOG"],
    "mexico": ["MEX", "CUN"],
    "asia": ["NRT", "HND", "ICN", "PEK", "PVG", "HKG", "SIN", "BKK", "KUL", "DOH", "DXB"],
    "sudeste_asiatico": ["SIN", "BKK", "KUL", "HAN", "SGN", "MNL", "CGK", "DPS", "HKT"],
    "oriente_medio": ["DXB", "DOH", "IST", "AUH"],
}


# Mapeamentos reversos comuns (IATA → nome de cidade)
IATA_NAMES = {
    "VIX": "Vitória", "GRU": "São Paulo (GRU)", "CGH": "São Paulo (CGH)",
    "VCP": "Campinas", "FLN": "Florianópolis", "CWB": "Curitiba",
    "POA": "Porto Alegre", "REC": "Recife", "SSA": "Salvador",
    "FOR": "Fortaleza", "NAT": "Natal", "MCZ": "Maceió",
    "GIG": "Rio (GIG)", "SDU": "Rio (SDU)", "BSB": "Brasília",
    "CNF": "Belo Horizonte", "BEL": "Belém", "MAO": "Manaus",
    "CGB": "Cuiabá", "CGR": "Campo Grande", "GYN": "Goiânia",
    "JPA": "João Pessoa", "AJU": "Aracaju", "SLZ": "São Luís",
    "THE": "Teresina", "LIS": "Lisboa", "OPO": "Porto",
    "MAD": "Madrid", "BCN": "Barcelona", "CDG": "Paris",
    "FCO": "Roma", "MIA": "Miami", "JFK": "Nova York (JFK)",
    "LAX": "Los Angeles", "MEX": "Cidade do México",
    "SCL": "Santiago", "EZE": "Buenos Aires", "MVD": "Montevidéu",
    "BOG": "Bogotá", "LIM": "Lima", "IGU": "Foz do Iguaçu",
    "NVT": "Navegantes", "JOI": "Joinville", "UDI": "Uberlândia",
    "STM": "Santarém", "RBR": "Rio Branco", "PVH": "Porto Velho",
    "BVB": "Boa Vista", "MCP": "Macapá", "MAB": "Marabá",
    "PMW": "Palmas", "EWR": "Nova York (EWR)", "LGA": "Nova York (LGA)",
    "SFO": "San Francisco", "ORD": "Chicago", "BOS": "Boston",
    "ATL": "Atlanta", "DFW": "Dallas", "IAH": "Houston",
    "SEA": "Seattle", "LAS": "Las Vegas", "MCO": "Orlando",
    "FLL": "Fort Lauderdale", "DCA": "Washington DC",
}


# Feriados BR — datas em que brasileiros viajam (para popular queries tipo
# "Carnaval 2027", "Natal", "Réveillon")
HOLIDAYS = {
    # Formato: "nome": {ano: (start, end)}. Start/end são inclusivos.
    "carnaval": {
        2026: ("2026-02-14", "2026-02-18"),  # Sáb → Qua cinzas
        2027: ("2027-02-06", "2027-02-10"),
        2028: ("2028-02-26", "2028-03-01"),
    },
    "semana_santa": {
        2026: ("2026-03-29", "2026-04-05"),
        2027: ("2027-03-21", "2027-03-28"),
    },
    "tiradentes": {
        2026: ("2026-04-18", "2026-04-21"),
        2027: ("2027-04-17", "2027-04-21"),
    },
    "corpus_christi": {
        2026: ("2026-06-04", "2026-06-07"),
        2027: ("2027-05-27", "2027-05-30"),
    },
    "sete_setembro": {
        2026: ("2026-09-05", "2026-09-07"),
        2027: ("2027-09-04", "2027-09-07"),
    },
    "nossa_senhora": {
        2026: ("2026-10-10", "2026-10-12"),
        2027: ("2027-10-10", "2027-10-12"),
    },
    "finados": {
        2026: ("2026-10-31", "2026-11-02"),
        2027: ("2027-10-30", "2027-11-02"),
    },
    "proclamacao_republica": {
        2026: ("2026-11-14", "2026-11-15"),
        2027: ("2027-11-13", "2027-11-15"),
    },
    "natal": {
        2026: ("2026-12-23", "2026-12-27"),
        2027: ("2027-12-23", "2027-12-27"),
    },
    "reveillon": {
        2026: ("2026-12-28", "2027-01-05"),
        2027: ("2027-12-28", "2028-01-05"),
    },
}


MONTHS_PT = {
    "janeiro": 1, "fevereiro": 2, "marco": 3, "março": 3, "abril": 4,
    "maio": 5, "junho": 6, "julho": 7, "agosto": 8,
    "setembro": 9, "outubro": 10, "novembro": 11, "dezembro": 12,
}


def region_to_iatas(region_name: str) -> list[str]:
    """Traduz nome de região em lista de IATAs. Case-insensitive."""
    key = region_name.lower().strip().replace(" ", "_")
    # Remove acentos básicos
    key = key.replace("ó", "o").replace("ú", "u").replace("á", "a").replace("ê", "e").replace("ã", "a").replace("ç", "c")
    return REGIONS.get(key, [])


def holiday_date_range(name: str, year: int) -> Optional[tuple[str, str]]:
    """Retorna (start_iso, end_iso) pra um feriado+ano, ou None."""
    key = name.lower().strip().replace(" ", "_").replace("ó", "o").replace("ê", "e")
    return HOLIDAYS.get(key, {}).get(year)


def month_date_range(month_name: str, year: int) -> Optional[tuple[str, str]]:
    """Ex: 'julho', 2026 → ('2026-07-01', '2026-07-31')."""
    m = MONTHS_PT.get(month_name.lower().strip())
    if m is None:
        return None
    start = date(year, m, 1)
    # Último dia do mês
    if m == 12:
        end = date(year, 12, 31)
    else:
        end = date(year, m + 1, 1) - timedelta(days=1)
    return (start.isoformat(), end.isoformat())
