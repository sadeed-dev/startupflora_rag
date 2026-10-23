# Single date columns in India time

MySQL stores one ordinary IST date field for each timestamp: `created_at`,
`indexed_at`, `checked_at`, `touched`, `expires`, or `at`. There are no permanent
`*_epoch`, `*_utc`, or generated duplicate date fields. Existing and new rows use
India Standard Time (Asia/Kolkata, UTC+05:30), regardless of the server timezone.

Fields use DATETIME(6) to retain microseconds for cache expiry, leases, and
pagination. A value looks like `2026-10-23 12:43:14.887433`. For a report showing
whole seconds, use `DATE_FORMAT(created_at, '%Y-%m-%d %H:%i:%s')`.

## Update an existing server database

Upload the updated code. Stop all old API, ingestion and CLI writers, then take a
database backup. The migration drops the old helper columns only after copying
their dates into replacement fields and rebuilding affected indexes. MySQL DDL
commits implicitly, so a backup is necessary for a complete rollback.

```bash
cd ~/apps/startupflora_rag
pm2 stop startupflora-rag
# Take a database backup before continuing.
.venv/bin/python migrate_dates.py
.venv/bin/python migrate_dates.py --apply
pm2 restart startupflora-rag
```

The first invocation previews the migration without writes. `--apply` converts
either the original numeric/UTC schema or the paired generated/helper columns
introduced earlier. Temporary copy columns are removed by the final ALTER; no
helper fields remain. Interrupted copies can be resumed by rerunning the plan
and `--apply`, with all writers still stopped. Already converted date columns
are not shifted again.

Backend startup also ensures the migration is complete, so take the backup
**before starting the new code**. The database user needs CREATE, ALTER and CREATE
VIEW permissions. On large databases use the explicit migration command before
startup; it has a longer database timeout.

Check the result in Workbench, refreshing Tables and Views first:

```sql
SELECT created_at FROM messages ORDER BY created_at DESC LIMIT 10;
SELECT checked_at FROM http_cache ORDER BY checked_at DESC LIMIT 10;
SELECT created_at FROM rag_request_timings ORDER BY created_at DESC LIMIT 10;
```

UTC `2026-10-23 07:13:14` becomes IST `2026-10-23 12:43:14`, preserving its
actual moment. Do not add another 330-minute offset to these columns. All
timestamp-bearing RAG tables are covered. Tables without timestamps are not
assigned invented dates; date values inside stored JSON answers are unchanged.

## Application compatibility

The MySQL adapter converts the application's date parameters to IST datetimes
and converts returned dates to the existing API representations. Nondate numbers,
duration fields and counters stay numeric. Cache expiry, rate limits, locks,
conversation cursors and source metadata keep their existing behavior. SQLite
storage is unchanged. SQLite-to-MySQL verification compares timestamps at the
microsecond precision supported by MySQL.

External SQL writers should now write IST datetime values directly to the normal
date fields. Old application versions require the old schema; restore the database
backup together with the old code if rolling back.
