"""Single IST DATETIME fields in MySQL, with compatible application timestamps."""

import re
from datetime import datetime, timedelta, timezone

IST = timezone(timedelta(hours=5, minutes=30))
DATE_COLUMNS = {
    "documents": {"indexed_at": "epoch"},
    "chunks": {"indexed_at": "epoch"},
    "embedding_cache": {"created_at": "epoch"},
    "conversations": {"created_at": "epoch"},
    "messages": {"created_at": "epoch"},
    "leases": {"expires": "epoch"},
    "requests": {"at": "epoch"},
    "rate_locks": {"touched": "epoch"},
    "search_cache": {"created_at": "epoch"},
    "http_cache": {"checked_at": "epoch"},
    "verified_answer_cache": {"created_at": "epoch"},
    "rag_request_timings": {"created_at": "utc"},
    "rag_unanswered_questions": {"created_at": "utc"},
}


def generated_column(info):
    extra = info.get("EXTRA", "").upper()
    return "VIRTUAL GENERATED" in extra or "STORED GENERATED" in extra


def inspect_columns(db):
    rows = db.execute(
        "SELECT TABLE_NAME,COLUMN_NAME,DATA_TYPE,DATETIME_PRECISION,EXTRA "
        "FROM information_schema.COLUMNS WHERE TABLE_SCHEMA=DATABASE() "
        "ORDER BY TABLE_NAME,ORDINAL_POSITION"
    ).fetchall()
    return {(r["TABLE_NAME"], r["COLUMN_NAME"]): r for r in rows if "TABLE_NAME" in r}


def inspect_indexes(db):
    return db.execute(
        "SELECT TABLE_NAME,INDEX_NAME,NON_UNIQUE,SEQ_IN_INDEX,COLUMN_NAME,SUB_PART,COLLATION "
        "FROM information_schema.STATISTICS WHERE TABLE_SCHEMA=DATABASE() "
        "ORDER BY TABLE_NAME,INDEX_NAME,SEQ_IN_INDEX"
    ).fetchall()


def date_aliases(columns):
    """Identify actual IST fields; the name is retained for the store interface."""
    return {
        table: {column: kind for column, kind in fields.items()
                if (table, column) in columns
                and columns[table, column]["DATA_TYPE"].lower() == "datetime"
                and columns[table, column].get("DATETIME_PRECISION") == 6
                and not generated_column(columns[table, column])}
        for table, fields in DATE_COLUMNS.items()
    }


def migration_statements(columns, indexes=()):
    """Materialize dates before dropping old generated and backing fields.

    Temporary columns make interrupted copies resumable. Each final ALTER is
    atomic per table and rebuilds affected indexes on the original date names.
    DATETIME(6) preserves the precision needed by leases and keyset pagination.
    """
    statements = []
    for table, fields in DATE_COLUMNS.items():
        for column, kind in fields.items():
            original = columns.get((table, column))
            if original is None:
                continue
            backing = f"{column}_{kind}"
            backed = (table, backing) in columns
            generated = generated_column(original)
            if not backed and not generated and original["DATA_TYPE"].lower() == "datetime" and original.get("DATETIME_PRECISION") == 6:
                continue
            if backed and not generated:
                raise ValueError(f"Unexpected duplicate date columns in {table}.{column}")
            source = backing if backed else column
            info = columns[table, source]
            source_type = info["DATA_TYPE"].lower()
            if kind == "epoch" and source_type != "double":
                raise ValueError(f"Unexpected date type in {table}.{source}")
            if kind == "utc" and source_type not in {"datetime", "timestamp"}:
                raise ValueError(f"Unexpected date type in {table}.{source}")
            temporary = f"__migrate_{column}_ist"
            if (table, temporary) not in columns:
                statements.append(f"ALTER TABLE `{table}` ADD COLUMN `{temporary}` DATETIME(6) NULL")
            raw = f"`{source}`"
            if kind == "epoch":
                raw = f"TIMESTAMPADD(MICROSECOND, CAST(ROUND({raw} * 1000000) AS SIGNED), '1970-01-01 00:00:00')"
            statements.append(f"UPDATE `{table}` SET `{temporary}`=DATE_ADD({raw}, INTERVAL 330 MINUTE)")
            affected = {}
            for index in indexes:
                if index.get("TABLE_NAME") == table:
                    affected.setdefault(index["INDEX_NAME"], []).append(index)
            changes, additions = [], []
            for name, parts in affected.items():
                if not any(part["COLUMN_NAME"] in {column, backing} for part in parts):
                    continue
                if name == "PRIMARY":
                    raise ValueError(f"Unexpected date primary key on {table}")
                changes.append(f"DROP INDEX `{name}`")
                terms = []
                for part in parts:
                    field = column if part["COLUMN_NAME"] == backing else part["COLUMN_NAME"]
                    term = f"`{field}`"
                    if part.get("SUB_PART"):
                        term += f"({int(part['SUB_PART'])})"
                    if part.get("COLLATION") == "D":
                        term += " DESC"
                    terms.append(term)
                unique = "UNIQUE " if not parts[0]["NON_UNIQUE"] else ""
                additions.append(f"ADD {unique}INDEX `{name}` ({','.join(terms)})")
            changes.append(f"DROP COLUMN `{column}`")
            default = " DEFAULT (DATE_ADD(UTC_TIMESTAMP(6), INTERVAL 330 MINUTE))" if kind == "utc" else ""
            changes.append(f"CHANGE COLUMN `{temporary}` `{column}` DATETIME(6) NOT NULL{default}")
            if backed:
                changes.append(f"DROP COLUMN `{backing}`")
            statements.append(f"ALTER TABLE `{table}` " + ", ".join(changes + additions))
    return statements


def migrate_dates(db):
    """Stop writers and back up before calling; reruns do not shift dates again."""
    db.execute("SET SESSION time_zone='+00:00'")
    statements = migration_statements(inspect_columns(db), inspect_indexes(db))
    for statement in statements:
        db.execute(statement)
    return statements


def refresh_report_views(db, columns):
    for table, fields in DATE_COLUMNS.items():
        if not all((table, name) in columns for name in fields):
            continue
        selected = [f"`{name}`" for owner, name in columns if owner == table]
        db.execute(f"CREATE OR REPLACE VIEW `{table}_report` AS SELECT {','.join(selected)} FROM `{table}`")
    if ("messages", "created_at") in columns:
        db.execute("CREATE OR REPLACE VIEW chat_messages_report AS SELECT m.conversation_id, "
                   "c.user_id AS visitor_id,m.seq,m.role,m.content,m.created_at AS createdAt "
                   "FROM messages m JOIN conversations c ON c.id=m.conversation_id")
    if ("conversations", "created_at") in columns:
        db.execute("CREATE OR REPLACE VIEW chat_conversations_report AS SELECT id,user_id, "
                   "created_at AS createdAt FROM conversations")


_LITERAL = re.compile(r"'(?:''|\\.|[^'\\])*'")
_IDENTIFIER = r"(?:`?\w+`?\.)?`?(\w+)`?"


def query_dates(sql, aliases):
    masked = _LITERAL.sub(lambda m: " " * len(m.group()), sql)
    if not re.match(r"\s*(SELECT|INSERT|UPDATE|DELETE|REPLACE)\b", masked, re.I):
        return {}, masked
    mapping = {}
    for match in re.finditer(r"\b(?:FROM|JOIN|INTO|UPDATE)\s+`?(\w+)`?", masked, re.I):
        mapping.update(aliases.get(match[1].lower(), {}))
    return mapping, masked


def to_storage(value, kind):
    if value is None:
        return None
    if kind == "epoch":
        return datetime.fromtimestamp(float(value), IST).replace(tzinfo=None)
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    return value.replace(tzinfo=value.tzinfo or timezone.utc).astimezone(IST).replace(tzinfo=None)


def application_parameters(sql, params, aliases):
    """Convert only parameters bound to known date fields; keep other numbers intact."""
    mapping, masked = query_dates(sql, aliases)
    if not mapping or not params:
        return params
    positions = [m.start() for m in re.finditer(r"%s", masked)]
    converted = list(params)
    kinds = {}
    insert = re.search(r"\b(?:INSERT|REPLACE)\s+INTO\s+`?\w+`?\s*\(([^)]+)\)\s*VALUES\s*\(([^)]+)\)", masked, re.I)
    if insert:
        columns = [field.strip().strip('`').lower() for field in insert[1].split(',')]
        values = list(re.finditer(r"%s", insert[2]))
        if len(values) == len(columns):
            for column, match in zip(columns, values):
                if column in mapping:
                    kinds[positions.index(insert.start(2) + match.start())] = mapping[column]
    for match in re.finditer(_IDENTIFIER + r"\s*(?:>=|<=|=|>|<)\s*(%s)", masked, re.I):
        if match[1].lower() in mapping:
            kinds[positions.index(match.start(2))] = mapping[match[1].lower()]
    for match in re.finditer(r"\(([^()]+)\)\s*(?:>=|<=|>|<)\s*\(([^()]+)\)", masked):
        columns = [field.strip().strip('`').lower() for field in match[1].split(',')]
        values = list(re.finditer(r"%s", match[2]))
        if len(columns) == len(values):
            for column, value in zip(columns, values):
                if column in mapping:
                    kinds[positions.index(match.start(2) + value.start())] = mapping[column]
    for index, kind in kinds.items():
        converted[index] = to_storage(converted[index], kind)
    return tuple(converted)


def application_row(row, mapping):
    result = dict(row)
    for column, kind in mapping.items():
        value = result.get(column)
        if isinstance(value, datetime):
            aware = value.replace(tzinfo=IST)
            result[column] = aware.timestamp() if kind == "epoch" else aware.astimezone(timezone.utc).replace(tzinfo=None)
    return result
