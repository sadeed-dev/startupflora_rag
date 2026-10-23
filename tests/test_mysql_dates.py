"""Single IST date migration and application compatibility without remote services."""

from datetime import datetime, timezone

import pytest

from rag.mysql_dates import (
    DATE_COLUMNS, IST, application_parameters, application_row, date_aliases,
    migration_statements, to_storage,
)


def metadata(layout="simple"):
    result = {}
    for table, fields in DATE_COLUMNS.items():
        for column, kind in fields.items():
            result[table, column] = {
                "DATA_TYPE": "datetime" if layout != "legacy" or kind == "utc" else "double",
                "DATETIME_PRECISION": 6 if layout == "simple" else 0,
                "EXTRA": "VIRTUAL GENERATED" if layout == "paired" else "",
            }
            if layout == "paired":
                result[table, f"{column}_{kind}"] = {
                    "DATA_TYPE": "double" if kind == "epoch" else "datetime", "EXTRA": "",
                }
    return result


@pytest.mark.parametrize("layout", ["legacy", "paired"])
def test_migration_copies_dates_before_removing_old_columns(layout):
    plan = migration_statements(metadata(layout))
    assert len(plan) == len(DATE_COLUMNS) * 3
    for i in range(0, len(plan), 3):
        assert "ADD COLUMN `__migrate_" in plan[i]
        assert plan[i + 1].startswith("UPDATE") and "330 MINUTE" in plan[i + 1]
        assert "CHANGE COLUMN" in plan[i + 2] and "DATETIME(6) NOT NULL" in plan[i + 2]
        assert "DROP COLUMN" in plan[i + 2]
        assert "GENERATED ALWAYS" not in plan[i + 2]
    if layout == "paired":
        assert any("DROP COLUMN `created_at_epoch`" in sql for sql in plan)
        assert any("DROP COLUMN `created_at_utc`" in sql for sql in plan)
    assert migration_statements(metadata("simple")) == []


def test_affected_indexes_are_restored_on_the_single_date_column():
    indexes = [
        {"TABLE_NAME": "conversations", "INDEX_NAME": "conversations_user_created", "NON_UNIQUE": 1, "COLUMN_NAME": column}
        for column in ("user_id", "created_at_epoch", "id")
    ]
    plan = migration_statements(metadata("paired"), indexes)
    sql = next(sql for sql in plan if "DROP COLUMN `created_at_epoch`" in sql and "conversations" in sql)
    assert "DROP INDEX `conversations_user_created`" in sql
    assert "ADD INDEX `conversations_user_created` (`user_id`,`created_at`,`id`)" in sql


def test_interrupted_copy_reuses_its_temporary_column():
    columns = metadata("paired")
    columns["messages", "__migrate_created_at_ist"] = {"DATA_TYPE": "datetime", "EXTRA": ""}
    plan = migration_statements(columns)
    assert not any("ALTER TABLE `messages` ADD COLUMN" in sql for sql in plan)
    assert any("UPDATE `messages`" in sql for sql in plan)


def test_rejects_ambiguous_manual_conversion():
    columns = metadata("legacy")
    columns["messages", "created_at"]["DATA_TYPE"] = "datetime"
    with pytest.raises(ValueError, match="Unexpected date type"):
        migration_statements(columns)


def test_date_binding_keeps_sql_names_and_nondate_numbers_unchanged():
    aliases = date_aliases(metadata())
    epoch = datetime(2026, 10, 23, 7, 13, 14, 887433, tzinfo=timezone.utc).timestamp()
    sql = "INSERT INTO messages (conversation_id,seq,role,content,created_at,metadata) VALUES (%s,%s,%s,%s,%s,%s)"
    bound = application_parameters(sql, ("cid", 2, "user", "created_at", epoch, "{}"), aliases)
    assert bound == ("cid", 2, "user", "created_at", datetime(2026, 10, 23, 12, 43, 14, 887433), "{}")
    query = "SELECT * FROM messages WHERE created_at>%s AND seq>%s AND content='created_at'"
    assert application_parameters(query, (epoch, 2), aliases) == (bound[4], 2)
    cursor = "SELECT id,created_at FROM conversations WHERE user_id=%s AND (created_at,id)<(%s,%s) LIMIT %s"
    assert application_parameters(cursor, ("alice", epoch, "id", 20), aliases) == ("alice", bound[4], "id", 20)
    assert application_parameters("SELECT * FROM schema_meta WHERE value=%s", (epoch,), aliases) == (epoch,)


def test_utc_timing_values_are_stored_once_as_ist():
    aliases = date_aliases(metadata())
    sql = "INSERT INTO rag_request_timings (request_id,created_at,total_ms) VALUES (%s,%s,%s)"
    bound = application_parameters(sql, ("rid", "2026-10-23 07:13:14", 100.5), aliases)
    assert bound == ("rid", datetime(2026, 10, 23, 12, 43, 14), 100.5)
    result = application_row({"created_at": bound[1], "total_ms": 100.5}, {"created_at": "utc"})
    assert result == {"created_at": datetime(2026, 10, 23, 7, 13, 14), "total_ms": 100.5}


def test_application_reads_single_dates_as_compatible_numeric_timestamps():
    date = datetime(2026, 10, 23, 12, 43, 14, 887433)
    result = application_row({"content": "Hello", "created_at": date, "metadata": "{}"}, {"created_at": "epoch"})
    assert list(result) == ["content", "created_at", "metadata"]
    assert to_storage(result["created_at"], "epoch") == date
    assert application_row({"count": 100}, {"created_at": "epoch"}) == {"count": 100}


def test_default_expression_is_not_mistaken_for_a_generated_date_column():
    columns = metadata()
    columns["rag_unanswered_questions", "created_at"]["EXTRA"] = "DEFAULT_GENERATED"
    assert migration_statements(columns) == []
    assert date_aliases(columns)["rag_unanswered_questions"] == {"created_at": "utc"}


def test_sqlite_migration_comparison_uses_microsecond_precision():
    from migrate_sqlite_to_mysql import normalized_row

    value = 1792739594.8874333
    expected = to_storage(value, "epoch").replace(tzinfo=IST).timestamp()
    assert normalized_row("messages", ("content", "created_at"), ("Hello", value)) == ("Hello", expected)
