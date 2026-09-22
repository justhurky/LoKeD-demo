# HOK AI Demo

Lokális, dokumentumalapú HÖK/EHÖK adminisztrációs asszisztens demo. A projekt célja, hogy helyi dokumentumokból forrásalapon válaszoljon, meetingjegyzeteket elemezzen, feladatokat és döntéseket emeljen ki, valamint egy egyszerű webes felületen kezelje a HÖK-admin munkafolyamatokat.

Ez a repository publikus demónak van tisztítva. Nem tartalmaz valós HÖK/EHÖK dokumentumokat, adatbázist, feltöltött fájlokat, modellcheckpointeket, API kulcsokat vagy személyes adatokat.

## Fő funkciók

- lokális FastAPI backend és egyoldalas web UI,
- Gemini-szerű fájlfeltöltés a chatben,
- dokumentum-indexelés és keresés,
- forrásalapú lokális RAG chat,
- meetingjegyzet elemzés,
- feladatlista kezelés,
- döntések és War Room áttekintés,
- `.txt`, `.md`, `.csv`, `.docx`, `.docm`, `.pdf`, `.xlsx`, `.xlsm` dokumentumok olvasása,
- offline-first működés Ollama modellekkel.

## Adatvédelem

Alapértelmezés szerint a demo lokálisan működik. A feltöltött dokumentumok a futtató gépen létrejövő `hok_ai/data/` mappába kerülnek, amely gitből ki van zárva.

Publikus GitHubra ne tölts fel:

- `hok_ai/data/`,
- `data/`,
- `.env`,
- valódi jegyzőkönyveket,
- valós feladatlistákat,
- modellcheckpointeket,
- API kulcsokat,
- tanúsítványokat.

## Gyors indítás

Python 3.11+ ajánlott.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Indítás:

```powershell
python main.py --server-only
```

Ezután nyisd meg:

```text
http://127.0.0.1:7842
```

## Ollama modell

Telepített Ollama mellett húzz le egy kisebb, demóhoz elég modellt:

```powershell
ollama pull qwen2.5:7b-instruct
ollama pull nomic-embed-text
```

Ha gyengébb gépen futtatod:

```powershell
ollama pull gemma3:4b
```

A demo alapbeállítása:

- chat modell: `qwen2.5:7b-instruct`,
- fallback modell: `gemma3:4b`,
- embedding modell: `nomic-embed-text`,
- PEFT adapter: kikapcsolva.

## Demo dokumentumok

A `demo_docs/` mappa fiktív mintákat tartalmaz. Ezekkel kipróbálható:

- dokumentum-indexelés,
- chat fájlfeltöltés,
- meeting elemzés,
- feladatok kinyerése.

## Tesztek

```powershell
pytest -q
```

## GitHub publikálási szabály

Ez a demo repository csak tisztított kódot és fiktív mintákat tartalmazzon. Valós EHÖK/HÖK anyag, adatbázis, feltanított adapter és lokális feltöltés ne kerüljön bele.
