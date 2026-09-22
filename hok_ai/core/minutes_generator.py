from __future__ import annotations

from dataclasses import dataclass

from .llm import PRIMARY_MODEL, OllamaClient


HOK_MINUTES_SYSTEM_PROMPT = """Te egy rendkivul preciz, logikus es profi Hallgatoi Onkormanyzati (HOK) adminisztracios AI asszisztens vagy. A feladatod, hogy a hallgatoi ulesekrol keszult nyers, beszelt nyelvi, sokszor csapongo es toltelekszavakkal teli leiratokbol tiszta, hivatalos dokumentaciot keszits.

A munkad soran az alabbi szabalyokat kell kovetned:

1. Szurd ki a toltelekszavakat, peldaul: "ooo", "hat", "igazabol", es a felesleges ismetleseket.
2. Keszits vilagos, formazott jegyzokonyvet a megvitatott napirendi pontokbol es dontesekbol.
3. Gyujtsd ki a konkret feladatokat, felelosoket es hataridoket.
4. Az elhangzottak alapjan tegyel proaktiv, vezetoi szintu javaslatokat a szervezet jovobeli mukodesere vagy a felmerult problemak megoldasara.
5. Ne talalj ki felelos, hatarido, szavazasi eredmeny vagy dontes adatot. Ha nem ismert, ird azt, hogy "Nem ismert".
6. Mindig magyarul valaszolj.
"""


HOK_MINUTES_USER_TEMPLATE = """Kerlek, dolgozd fel az alabbi nyers HOK ules leiratot, es keszitsd el a hivatalos dokumentaciot az alabbi struktura szerint.

ELVART STRUKTURA:

1. Rovid Osszefoglalo:
(2-3 mondatban foglald ossze az ules legfontosabb temait es hangvetelet.)

2. Hivatalos Jegyzokonyv:
(Napirendi pontokra bontva ird le, hogy mirol volt szo, milyen ervek hangzottak el, es milyen dontes vagy szavazasi eredmeny szuletett.)

3. Feladatnaplo (Teendok):
(Szigoruan csak az elvegzendo feladatokat sorold fel az alabbi formatumban:

[Felelos neve]: [Feladat rovid leirasa] - Hatarido: [Datum vagy "Nem ismert"])

4. Strategiai Javaslatok:
(Az ulesen elhangzott problemak, feszultsegek vagy otletek alapjan irj 1-2 konstruktiv javaslatot arra vonatkozoan, hogyan lephetne tovabb a szervezet, vagy hogyan lehetne hatekonyabb a mukodes.)

NYERS LEIRAT:
{transcript}
"""


@dataclass(slots=True)
class MinutesGenerator:
    llm: OllamaClient
    model: str = PRIMARY_MODEL

    def generate(self, transcript: str, *, timeout: int = 180) -> str:
        if not transcript.strip():
            raise ValueError("Jegyzokonyv generalashoz nyers leirat szukseges.")
        messages = [
            {"role": "system", "content": HOK_MINUTES_SYSTEM_PROMPT},
            {"role": "user", "content": HOK_MINUTES_USER_TEMPLATE.format(transcript=transcript.strip())},
        ]
        return str(
            self.llm.chat(
                messages,
                model=self.model,
                stream=False,
                think=False,
                timeout=timeout,
                options={"temperature": 0.15, "num_predict": 2600},
            )
        ).strip()
