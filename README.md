# LoKeD Demo

Lokalis, dokumentumalapu adminisztracios asszisztens demo. A projekt celja, hogy egy adott munkakornyezetben hasznos, forrasalapu AI asszisztenst adjon: dokumentumokat indexel, ezekbol valaszol, meetingjegyzeteket elemez, feladatokat es donteseket emel ki, majd egy egyszeru webes feluleten tamogatja az adminisztrativ munkat.

Ez a repository publikus demonak van tisztitva. Nem tartalmaz valos dokumentumokat, adatbazist, feltoltott fajlokat, modellcheckpointeket, API kulcsokat vagy szemelyes adatokat.

## Tartalom

- [Roviden](#roviden)
- [Mit tud a demo?](#mit-tud-a-demo)
- [Mukodesi elv](#mukodesi-elv)
- [Technologiai stack](#technologiai-stack)
- [Mappaszerkezet](#mappaszerkezet)
- [Telepites](#telepites)
- [Ollama modellek](#ollama-modellek)
- [Inditas](#inditas)
- [Hasznalat](#hasznalat)
- [API vegpontok](#api-vegpontok)
- [Adatvedelem](#adatvedelem)
- [Tesztek](#tesztek)
- [Hibakereses](#hibakereses)
- [GitHub publikasi checklist](#github-publikasi-checklist)
- [Fejlesztesi roadmap](#fejlesztesi-roadmap)

## Roviden

A HOK AI Demo egy offline-first RAG asszisztens. A RAG jelentese: Retrieval-Augmented Generation, vagyis a modell nem csak altalanos tudasbol valaszol, hanem eloszor keres a lokalis dokumentumtarban, majd a megtalalt forrasreszletekre epiti a valaszt.

Tipikus feladatok:

- "Milyen dontes szuletett a meetingben?"
- "Ki felelos a technikai arajanlatokert?"
- "Foglaldd ossze ezt a feltoltott DOCX fajlt."
- "Ebbol a jegyzetbol keszits feladatlistat."
- "Mely nyitott feladatoknak nincs hatarideje?"

## Mit tud a demo?

### Chat asszisztens

Termeszetes nyelven lehet kerdezni az indexelt dokumentumokrol. A valasz mellett a rendszer visszaadja a forrasokat is, igy ellenorizheto, mibol dolgozott.

### Gemini-szeru fajlfeltoltes

A Chat oldalon fajlt lehet csatolni, majd melle konkret utasitast adni.

Peldak:

- "Foglaldd ossze hivatalos stilusban."
- "Javitsd ki es tedd jegyzokonyv formatumba."
- "Emeld ki belole a feladatokat, felelosoket es hataridoket."
- "Mondd meg, milyen tablazat ez, es milyen mezok vannak benne."

Tamogatott fajltipusok:

- `.txt`
- `.md`
- `.csv`
- `.docx`
- `.docm`
- `.pdf`
- `.xlsx`
- `.xlsm`

### Dokumentum-indexeles

A Dokumentumok oldalon fajl vagy teljes mappa indexelheto. Az indexelt tartalom bekerul a lokalis SQLite adatbazisba es kereshetove valik.

### Tudastar kereses

Az app teljes szoveges keresest biztosit a betoltott dokumentumokon. A talalatok tartalmazzak a dokumentum nevet, utvonalat, tipusat es relevans reszletet.

### Meeting elemzes

Nyers meetingjegyzetbol strukturalt elemzest keszit:

- feladatok,
- felelosok,
- hataridok,
- dontesek,
- kerdesek,
- blokkolok,
- kaosz-index.

### Feladatlista

A meeting elemzesbol kinyert vagy kezzel rogzitett feladatokat lokalis adatbazisban tarolja. A feladatoknal kezelheto:

- cim,
- leiras,
- felelos,
- hatarido,
- prioritas,
- statusz,
- forrasmondat.

### War Room attekintes

Vezetoi osszefoglalo nezet, amely segit gyorsan latni:

- hany nyitott feladat van,
- mely feladatoknak nincs felelose,
- mely feladatoknak nincs hatarideje,
- milyen dontesek es blokkolok jelentek meg.

### Opcionalis training eszkozok

A `hok_ai/training/` mappa tartalmaz kiserleti eszkozoket dataset epiteshez es PEFT adapter futtatashoz. A publikus demoban nincs feltanitott modell vagy checkpoint. A default mukodes egyszeru lokalis Ollama modellre epul.

## Mukodesi elv

```text
Dokumentumok / jegyzetek / tablazatok
        |
        v
Dokumentum loader
        |
        v
Chunkolas es lokalis indexeles
        |
        v
SQLite tudastar
        |
        v
Kereses a relevans reszletekben
        |
        v
Ollama chat modell
        |
        v
Forrasalapu magyar valasz
```

Az app nem kuldi automatikusan felhobe a dokumentumokat. A lokalis demohoz Ollama szukseges.

## Technologiai stack

- Python 3.11+
- FastAPI
- Uvicorn
- SQLite
- ChromaDB opcionallis vektoros komponensekhez
- Ollama lokalis LLM futtatashoz
- vanilla HTML/CSS/JavaScript frontend
- `python-docx` DOCX olvasashoz
- `openpyxl` XLSX/XLSM olvasashoz
- `pdfplumber` PDF olvasashoz
- `pytest` tesztekhez

## Mappaszerkezet

```text
HOK-AI-demo/
  demo_docs/                 Fiktiv demo dokumentumok
  hok_ai/
    api/                     FastAPI route-ok
    core/                    Adatbazis, dokumentumolvasas, RAG, LLM kliens
    frontend/                Egyoldalas web UI
    training/                Opcionalis training es dataset eszkozok
    main.py                  Alkalmazas belepesi pont
  tests/                     Alap tesztek
  tools/                     Lokalis modell setup script
  main.py                    Gyoker belepesi pont
  requirements.txt           Fuggosegek
  README.md                  Ez a dokumentacio
```

Futas kozben letrejovo, nem verziozott mappak:

```text
hok_ai/data/
data/
.pytest_cache/
.pytest_tmp/
```

Ezek szandekosan szerepelnek a `.gitignore` fajlban.

## Telepites

### 1. Repository klonozasa

```powershell
git clone https://github.com/<felhasznalo>/HOK-AI-demo.git
cd HOK-AI-demo
```

### 2. Virtualis kornyezet

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Fuggosegek telepitese

```powershell
pip install -r requirements.txt
```

Ha regebbi Python vagy rendszerfuggosegi gond van, erdemes eloszor frissiteni:

```powershell
python -m pip install --upgrade pip setuptools wheel
```

## Ollama modellek

Az app lokalis LLM-et hasznal. Ehhez telepiteni kell az Ollamat:

https://ollama.com/

Ajanlott modellek:

```powershell
ollama pull qwen2.5:7b-instruct
ollama pull nomic-embed-text
```

Gyengebb gepre:

```powershell
ollama pull gemma3:4b
```

Alapertelmezett demo beallitas:

- chat modell: `qwen2.5:7b-instruct`
- fallback modell: `gemma3:4b`
- fajlfeldolgozo modell: `qwen2.5:7b-instruct`
- embedding modell: `nomic-embed-text`
- PEFT adapter: kikapcsolva

Ha a `qwen2.5:7b-instruct` tul nagy a gepnek, a Beallitasok oldalon allitsd a chat es fajl modellt `gemma3:4b` ertekre.

## Inditas

Szerver modban:

```powershell
python main.py --server-only
```

Alap URL:

```text
http://127.0.0.1:7842
```

Mas porton:

```powershell
python main.py --server-only --port 7850
```

Ha telepitve van a PyWebView, ablakos modban is indithato:

```powershell
python main.py
```

## Hasznalat

### 1. Demo dokumentumok indexelese

1. Inditsd el az appot.
2. Nyisd meg a `Dokumentumok` oldalt.
3. Add meg a `demo_docs` mappa utvonalat.
4. Inditsd az indexelest.
5. A Chat oldalon kerdezz ra a mintadokumentumokra.

Peldakerdesek:

```text
Milyen feladatok vannak a mintameetingben?
```

```text
Ki felelos a technikai arajanlatokert?
```

```text
Milyen dontes szuletett a beszerzesrol?
```

### 2. Fajl feltoltese a chatben

1. Menj a `Chat` oldalra.
2. Kattints a `Fajl` gombra.
3. Valassz ki egy tamogatott fajlt.
4. Irj melle utasitast.
5. Kattints a `Kuldes` gombra.

Pelda utasitas:

```text
Foglaldd ossze ezt a fajlt hivatalos HOK jegyzokonyv stilusban, es emeld ki a feladatokat.
```

### 3. Meeting elemzes

1. Menj a `Meeting elemzes` oldalra.
2. Illeszd be a nyers jegyzetet.
3. Inditsd az elemzest.
4. Ellenorizd a kinyert feladatokat, donteseket es blokkolokat.

### 4. Feladatok kezelese

A `Feladatok` oldalon attekinthetok es kezelhetok a rogzitett teendok. A cel az, hogy a meetingbol kijovo adminisztrativ terhek egy helyen kovethetok legyenek.

### 5. Beallitasok

A `Beallitasok` oldalon modosithato:

- Ollama host,
- chat modell,
- fallback modell,
- fajlfeldolgozo modell,
- embedding modell,
- PEFT adapter opciok.

## API vegpontok

Fobb vegpontok:

```text
GET    /api/health
GET    /api/settings
PUT    /api/settings
POST   /api/chat
POST   /api/chat/stream
POST   /api/chat/file
GET    /api/documents
POST   /api/documents/upload
POST   /api/documents/index-path
GET    /api/documents/search
POST   /api/meeting/analyze
GET    /api/tasks
POST   /api/tasks
PUT    /api/tasks/{task_id}
DELETE /api/tasks/{task_id}
GET    /api/dashboard
```

Egyszeru health check:

```powershell
Invoke-RestMethod http://127.0.0.1:7842/api/health
```

Fajl feltoltes pelda `curl` paranccsal:

```powershell
curl.exe -X POST "http://127.0.0.1:7842/api/chat/file" `
  -F "instruction=Foglaldd ossze roviden ezt a fajlt." `
  -F "index=true" `
  -F "file=@demo_docs/sample_meeting.md"
```

## Adatvedelem

Ez a demo ugy van kialakitva, hogy publikus GitHub repositoryban is vallalhato legyen. Ennek ellenere a futtatas kozben keletkezo adatokat nem szabad commitolni.

Ne commitold:

- `hok_ai/data/`
- `data/`
- `.env`
- valodi jegyzokonyvek
- valos feladatlistak
- hallgatoi adatok
- szemelyes adatok
- API kulcsok
- tanusitvanyok
- modellcheckpointek
- `.safetensors`, `.bin`, `.pt`, `.pth`, `.gguf` fajlok

A `.gitignore` ezeket alapbol tiltja, de publikalas elott mindig ellenorizd:

```powershell
git status --short --ignored
git diff --cached --name-only
```

Titokkereseshez:

```powershell
git grep -n -I "api_key\|password\|secret\|sk-\|AIza"
```

## Tesztek

Alap tesztek:

```powershell
pytest -q --basetemp .pytest_tmp
```

Ha Windows alatt a globalis temp konyvtar jogosultsagi hibat ad, mindig hasznalj helyi `--basetemp` erteket:

```powershell
python -m pytest -q --basetemp .pytest_tmp
```

## Hibakereses

### Nem indul az app

Ellenorizd, hogy a port szabad-e:

```powershell
netstat -ano | Select-String ":7842"
```

Masik port:

```powershell
python main.py --server-only --port 7850
```

### Az Ollama offline

Ellenorzes:

```powershell
ollama list
Invoke-RestMethod http://127.0.0.1:11434/api/tags
```

Ha nincs modell:

```powershell
ollama pull qwen2.5:7b-instruct
ollama pull nomic-embed-text
```

### A nagy modell memoriahibaval leall

Allitsd a modellt kisebbre:

```text
gemma3:4b
```

Ezt az app Beallitasok oldalan tudod megtenni.

### PDF olvasasi hiba

Gyozodj meg rola, hogy a fuggosegek telepitve vannak:

```powershell
pip install -r requirements.txt
```

### DOCX vagy XLSX olvasasi hiba

Ellenorizd:

```powershell
pip show python-docx
pip show openpyxl
```

## GitHub publikasi checklist

Publikalas elott:

```powershell
git status --short
git diff --cached --name-only
python -m pytest -q --basetemp .pytest_tmp
```

Ellenorizd, hogy nincs staged allapotban:

- adatbazis,
- `.env`,
- feltoltott dokumentum,
- privat jegyzokonyv,
- modellcheckpoint,
- tanusitvany,
- release zip.

GitHub repo letrehozasa GitHub CLI-val:

```powershell
gh repo create HOK-AI-demo --public --source . --remote origin --push
```

Privat repo eseten:

```powershell
gh repo create HOK-AI-demo --private --source . --remote origin --push
```

## Fejlesztesi roadmap

Rovid tav:

- stabilabb meeting elemzes,
- szebb forrasmegjelenites,
- feladatok exportja CSV/JSON formatumban,
- dokumentumtipus-cimkek finomitasa,
- felhasznalobaratabb hibajelzesek.

Kozep tav:

- People Memory,
- Role Memory,
- tanulasi javaslatok,
- hangfajl import es Whisper feldolgozas,
- anonimizalt cloud mod opcionalis tamogatasa.

Hosszu tav:

- teljes lokalis tudastar,
- szerkesztheto dontestar,
- Excel sablonmemoria,
- auditalhato tanulasi naplo,
- tobbmodellu benchmark panel.

## Fontos megjegyzes

Ez demoprojekt, nem hivatalos jogi vagy szabalyzati donteshozo rendszer. A valaszokat es a kinyert feladatokat mindig embernek kell ellenoriznie, kulonosen szabalyzatertelmezes, szemelyes adatok es hivatalos dontesek eseten.

## Licenc

Licenc meg nincs megadva. Publikus GitHub hasznalat elott erdemes valasztani egy megfelelo licencet, peldaul MIT vagy Apache-2.0, ha nyilt forraskent szeretnenk kiadni.
