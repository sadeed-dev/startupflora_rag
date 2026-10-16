# Upgrade from the first plain-Python version

Use the whole v2 project. Replacing main.py alone will not migrate the model,
retrieval, tools or vector payload schema.

## Keep existing data safe

1. Keep your original project, .env and data directory as a backup.
2. Extract v2 into a fresh folder, for example `C:\projects\startupflora-rag-v2`.
3. Create a fresh virtualenv and install the new requirements.txt there.
4. Copy .env.example to .env and copy your real provider key into it. Add your
   backend key only if you are starting the protected API.
5. Keep v2 defaults initially:

```dotenv
QDRANT_MODE=local
QDRANT_PATH=data/v2/qdrant
DB_PATH=data/v2/rag.sqlite3
QDRANT_COLLECTION=startupflora_lc_v2
```

These names avoid overwriting the first version's corpus. Local file storage has
no database URL/dashboard and requires no Docker. In server mode explicitly set
`QDRANT_MODE=server`; QDRANT_URL is used only then. Use a new dedicated collection.

A QDRANT_PATH environment variable set earlier in PowerShell overrides .env.
Run `python main.py doctor` and verify the active paths before indexing. Start a
fresh PowerShell window or adjust that environment variable if it points at the
old data. Do not erase old directories to fix a configuration mismatch.

## Reindex the content

LangChain's QdrantVectorStore uses a document/metadata payload that differs from
v1. v2 also uses different chunking and collection namespace checks. Reindex into
the new corpus; pointing v2 at old vectors is not a supported shortcut.

```powershell
python main.py doctor
python main.py ingest-web --max-pages 5
python main.py ask "What startup services do you offer?"
python main.py ingest-web --max-pages 0
python main.py status
```

Review the crawl report: 0 means all discovered URLs up to the configured hard
cap, not a promise to find every hidden/JavaScript/private page. It does not mean
zero pages. Current scheme facts still require approved, current source content.

## Optionally import old conversation history

Your old app's history remains in its SQLite file. This command copies
non-conflicting conversations, their original owners/UUIDs, messages, summary and
source metadata; it imports no old corpus or answer retry cache:

```powershell
python main.py migrate-history --from-db C:\projects\startupflora-rag\data\rag.sqlite3
```

Adjust the path to the actual old DB. Existing destination conversation IDs are
left intact. The old DB is opened read-only using a consistent snapshot. Then
resume an imported conversation with its original user identity:

```powershell
python main.py chat --user ORIGINAL-USER-ID --conversation ORIGINAL-CONVERSATION-UUID
```

A new conversation starts fresh. Imported assistant replies are historical text,
not evidence for new answers; new replies retrieve from the newly indexed corpus.

## Connect the website after CLI testing

The private API still requires a 32+ character BACKEND_API_KEY. This is independent
of your OpenAI key. Keep it on your website backend/RAG server only. Reuse the
Express adapter pattern and adapt req.user.id to your existing authentication.
The route contracts for /chat and message history remain compatible; GET
/conversations now allows the website to list and resume owned chats.

Changing embedding model/dimensions, collection or Qdrant target later requires
a new corpus DB/collection and reindexing. You can migrate conversation history
again. Matching vector dimensions alone do not make embedding spaces compatible.
