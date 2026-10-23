"""Plan/apply readable IST dates without provider calls. Stop writers and back up first."""

import argparse
import json

import pymysql
from pymysql.cursors import DictCursor

from rag.config import load_settings
from rag.mysql_dates import inspect_columns, inspect_indexes, migrate_dates, migration_statements, refresh_report_views
from rag.mysql_store import _Session


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Apply the migration; default only prints the plan")
    args = parser.parse_args()
    settings = load_settings()
    if settings.db_backend != "mysql":
        parser.error("This migration is for DB_BACKEND=mysql")
    tls = {}
    if settings.mysql_ssl_ca:
        tls = dict(ssl_ca=settings.mysql_ssl_ca, ssl_verify_cert=True, ssl_verify_identity=True)
    try:
        with pymysql.connect(
            host=settings.mysql_host, port=settings.mysql_port, user=settings.mysql_user,
            password=settings.mysql_password, database=settings.mysql_database,
            cursorclass=DictCursor, charset="utf8mb4", connect_timeout=10,
            read_timeout=300, write_timeout=300, **tls,
        ) as connection:
            db = _Session(connection)
            if args.apply:
                statements = migrate_dates(db)
                db.execute("UPDATE schema_meta SET value='2' WHERE `key`='version'")
                refresh_report_views(db, inspect_columns(db))
                connection.commit()
            else:
                statements = migration_statements(inspect_columns(db), inspect_indexes(db))
            print(json.dumps({"applied": args.apply, "timezone": "Asia/Kolkata",
                              "statements": statements}, indent=2))
    except pymysql.Error as exc:
        raise SystemExit(f"MySQL date migration failed ({type(exc).__name__}, code {exc.args[0]}). Check server logs and database access; rerun the plan before retrying.") from None


if __name__ == "__main__":
    main()
