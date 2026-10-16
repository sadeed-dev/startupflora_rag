# MySQL patch for StartupFlora RAG (LangChain v2)

Yeh patch sirf relational storage change karta hai. Qdrant collection, vectors,
chunk IDs, crawler aur prompts same rehte hain. Python code mein function-purpose
docstrings/comments hain. Existing SQLite records migrate karoge toh embeddings
regenerate karna zaroori nahi.

## Files to copy

Stop chat/API/ingestion first. ZIP extract karke files **existing project root**
`C:\projects\startupflora-rag` mein merge/copy karo. Same names replace karo:

| File | Why it changed |
|---|---|
| `rag/config.py` | DB_BACKEND aur MYSQL_* settings |
| `rag/store.py` | SQLite/MySQL store selector; SQLite fallback preserved |
| `rag/runtime.py` | Selected store initialize karna |
| `main.py` | doctor/status/cache/history commands selected database use karein |
| `rag/mysql_store.py` (new) | MySQL tables, transactions, FULLTEXT and all store operations |
| `migrate_sqlite_to_mysql.py` (new) | Existing corpus/history/cache safely copy + verify |
| `requirements-mysql.txt` (new) | Only additional MySQL driver |

**Apna existing `.env` ya `data` folder replace/delete mat karo.** `rag/crawler.py`,
`rag/llm.py`, `rag/ingest.py`, API endpoints aur Qdrant configuration change nahi
karni. `MYSQL_SETUP.md` yeh instruction file hai; executable code nahi.

## 1. Install additional dependency

PowerShell, existing project environment:

```powershell
cd C:\projects\startupflora-rag
python -m pip install -r requirements-mysql.txt
```

Supported target: MySQL 8.0+/8.4, or MariaDB 10.11+, using InnoDB.

## 2. Create an EMPTY dedicated database

MySQL Workbench mein apne admin account se run karo. Existing insights_db ya
central_db use mat karo: RAG ke liye separate database create karo.

```sql
CREATE DATABASE startupflora_rag
  CHARACTER SET utf8mb4 COLLATE utf8mb4_bin;

-- Apna actual strong password daalo; example text ko password mat rakho.
CREATE USER 'rag_app'@'localhost' IDENTIFIED BY 'REPLACE_WITH_YOUR_PASSWORD';
GRANT SELECT, INSERT, UPDATE, DELETE, CREATE, ALTER, INDEX
  ON startupflora_rag.* TO 'rag_app'@'localhost';
```

Existing dedicated MySQL user hai toh usko suitable privileges de sakte ho.
Remote MySQL host use karte ho toh account host restriction server administrator
se apne app host ke according configure karao. Tables Python store automatically
create karega; saari CREATE TABLE queries manually chalani nahi hain.

## 3. Add MySQL connection to existing .env

Migration ke dauran **DB_BACKEND=sqlite** rakho. Source DB_PATH wahi file hai jisme
492 pages/2844 chunks saved the. Agar tumhara current path alag hai, wahi use karo.

```env
DB_BACKEND=sqlite
DB_PATH=data/v2/rag-server.sqlite3

MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_DATABASE=startupflora_rag
MYSQL_USER=rag_app
MYSQL_PASSWORD="YOUR_ACTUAL_PASSWORD"
# Remote server CA certificate ho toh optional verified TLS:
MYSQL_SSL_CA=
```

Existing OPENAI_API_KEY, QDRANT_MODE, QDRANT_URL, QDRANT_COLLECTION,
EMBEDDING_MODEL, EMBEDDING_DIMENSIONS etc. **exactly same** rakho. Existing .env
ko replace nahi, sirf fields add/update karo. Shell environment values override
.env, isliye stale DB_BACKEND/MYSQL_* PowerShell overrides ho toh clear karo.

## 4. Migrate existing SQLite data ONCE

All chat/API/ingestion processes stopped hone chahiye; migration app writes ko
concurrently merge nahi karti. Qdrant vectors move nahi honge: wahi server/
collection use hoga, isliye migration mein OpenAI calls bhi nahi hain.

```powershell
python migrate_sqlite_to_mysql.py --from-db data/v2/rag-server.sqlite3
```

Script:

1. SQLite source read-only consistent snapshot se read karta hai.
2. Existing namespace/model ko current Qdrant configuration se match karta hai.
3. Target RAG tables empty hain confirm karta hai; existing MySQL data overwrite
   ya merge nahi karta.
4. Documents, chunks, embedding cache, conversations, messages, HTTP cache,
   saved answers aur corpus metadata copy karta hai.
5. Har copied table ke row count aur content hash verify karta hai.
6. Sab correct ho toh transaction commit; error ho toh copied MySQL rows rollback.
   Initial empty table/schema creation MySQL DDL hai, isliye woh remain kar sakti hai.
7. Original SQLite file aur Qdrant unchanged rehte hain.

SQLite FTS shadow tables copy nahi hongi; MySQL FULLTEXT automatically build hota
hai. Search-result ranks reset hote hain kyunki FULLTEXT ranking FTS5 se alag hai.
Expired/active leases aur request-rate windows copy nahi hote. New target starts
without those transient operations. Script `.env` ko automatically edit nahi karta.

Agar target schema contains rows error aaye, same target par migration repeat mat
karo. Agar migration already successful thi, next step follow karo. Failed partial
data transaction normally rollback ho jaati hai; unrelated existing data ko delete
mat karo. New empty dedicated database choose karna safe hai.

## 5. Activate MySQL only after migration is complete

Existing .env mein sirf yeh line change karo:

```env
DB_BACKEND=mysql
```

Then:

```powershell
python main.py doctor
python main.py status
python main.py chat
```

Doctor output `db_backend: mysql` aur MySQL host/port/name show karega. Doctor
configuration show karta hai; MySQL connection test **status** command karta hai.
Status mein indexed-source/chunk counts previous corpus se match hone chahiye.

Existing Qdrant server run rehna chahiye for chat/ingestion. MySQL server bhi run
rehna chahiye. MySQL port 3306 hai, Qdrant 6333: dono alag services hain.

## 6. Verify in MySQL Workbench

```sql
USE startupflora_rag;
SHOW TABLES;
SELECT COUNT(*) AS indexed_pages FROM documents;
SELECT COUNT(*) AS active_chunks FROM chunks;
SELECT COUNT(*) AS cached_embeddings FROM embedding_cache;
SELECT role,content FROM messages ORDER BY created_at DESC LIMIT 10;
SELECT id,source,title,position,text FROM chunks LIMIT 5;
```

If previous counts abhi bhi 492/2844 hain, migration ke baad same counts expected
hain; source meanwhile changed ho toh us source ke actual counts match karo.
Qdrant's point count can include orphan vector generations until cleanup; MySQL
chunks table is the active chunk list used for evidence filtering.

## 7. Future ingestion

```powershell
python main.py ingest-web --max-pages 0
```

Successfully migrated fingerprints/HTTP state skip unchanged pages. Migrated exact
embedding cache helps avoid repeat embedding calls. New/changed inputs may call
OpenAI. No re-ingestion is required just to switch the relational database.

## Notes on keyword search and rollback

- MySQL InnoDB FULLTEXT replaces SQLite FTS5 BM25. Rankings/tokenization are not
  identical; MySQL stopwords/minimum token length apply. Qdrant semantic search,
  rank fusion and reranking still run in the existing retrieval pipeline.
- Full UTF-8/Hindi text is preserved. URL uniqueness uses a full-source hash,
  avoiding incorrect collisions caused by unique indexes on URL prefixes.
- Conversations retain owner checks, paired-message transactions, pagination and
  retry answers. Rate checks use per-user row locks; publication/chat leases have
  background heartbeat renewal.
- To return to SQLite: stop processes and set DB_BACKEND=sqlite. Original DB_PATH
  still exists. MySQL changes made after migration are NOT automatically copied
  back to SQLite; choosing SQLite at that point shows the older snapshot.
- Old standalone scripts that import Store(DB_PATH) explicitly still open SQLite.
  For selected-backend scripts use create_store(load_settings()). The patch's
  main.py/runtime.py already do that. Qdrant migration/viewer utilities are separate.

## Validation performed

- Existing RAG regression suite: **42 tests passed** with the patched configuration,
  runtime and CLI (SQLite fallback).
- Real **MySQL 8.0.46**: **9 integration tests passed**.
- Real **MariaDB 10.11.14**: **9 integration tests passed**.
- Integration coverage: schema creation, FULLTEXT query, Hindi round-trip, long
  distinct URLs, page/chunk CRUD, vector cache, history/ownership, pagination,
  idempotent answer retrieval, atomic paired turns, parallel rate checks/leases,
  SQLite full migration, namespace mismatch, refuse-overwrite and rollback.
- End-to-end ingestion → Qdrant retrieval → chat → saved history tested with
  in-memory Qdrant and fake models. No paid OpenAI calls were made. The test-only
  Qdrant payload-index warning is expected for its in-memory mode.
- Ruff lint/format and Python syntax checks passed.
- Your Windows Python 3.14 installation, MySQL server credentials/TLS setup and
  real website/provider network calls were not executed here.

Reference for FULLTEXT differences:
https://dev.mysql.com/doc/refman/8.4/en/fulltext-boolean.html
