"""Pool isolation and recovery checks without network access or provider calls."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pymysql
import pytest

from rag.config import Settings
from rag.mysql_store import MySQLStore


class Connection:
    def __init__(self):
        self.open = True
        self.statements = []
        self.commits = self.rollbacks = 0
        self.stale = self.broken = False

    def cursor(self):
        connection = self

        class Cursor:
            description = True
            rowcount = 1

            def __enter__(self):
                return self

            def __exit__(self, *_):
                pass

            def execute(self, sql, params=None):
                connection.statements.append(sql)

            def fetchall(self):
                return [{"value": "1"}]

        return Cursor()

    def ping(self, reconnect=False):
        assert reconnect is False
        if self.stale:
            raise pymysql.OperationalError(2006, "Disconnected")

    def commit(self):
        if self.broken:
            raise pymysql.OperationalError(2013, "Connection lost")
        self.commits += 1

    def rollback(self):
        if self.broken:
            raise pymysql.OperationalError(2013, "Connection lost")
        self.rollbacks += 1

    def close(self):
        self.open = False


@pytest.fixture
def pool(monkeypatch):
    connections = []

    def connect(**_):
        connection = Connection()
        connections.append(connection)
        return connection

    monkeypatch.setattr(pymysql, "connect", connect)
    store = MySQLStore(Settings(mysql_pool_size=2, mysql_pool_timeout=0.05))
    yield store, connections
    store.close()


def test_reuses_connections_and_initializes_session_once(pool):
    store, connections = pool
    for _ in range(3):
        with store.connect() as db:
            db.execute("SELECT 1")
    assert len(connections) == 1
    assert connections[0].commits == 4
    assert sum(sql.startswith("SET SESSION") for sql in connections[0].statements) == 4


def test_rolls_back_before_reusing_connection(pool):
    store, connections = pool
    with pytest.raises(RuntimeError, match="test failure"):
        with store.connect():
            raise RuntimeError("test failure")
    assert connections[0].rollbacks == 1
    with store.connect():
        pass
    assert len(connections) == 1


def test_exclusive_borrowers_and_pool_exhaustion(pool):
    store, connections = pool
    barrier = Barrier(3)

    def borrow():
        with store.connect() as db:
            barrier.wait(timeout=5)
            barrier.wait(timeout=5)
            return id(db.connection)

    with ThreadPoolExecutor(max_workers=2) as workers:
        futures = [workers.submit(borrow) for _ in range(2)]
        barrier.wait(timeout=5)
        try:
            with pytest.raises(ValueError, match="pool is busy"):
                with store.connect():
                    pass
        finally:
            barrier.wait(timeout=5)
        assert len({future.result() for future in futures}) == 2
    with store.connect():
        pass
    assert len(connections) == 2


def test_replaces_stale_socket_without_retrying_a_transaction(pool):
    store, connections = pool
    connections[0].stale = True
    with store.connect() as db:
        assert db.connection is connections[1]
    assert len(connections) == 2
    assert not connections[0].open


def test_discards_failed_transaction_connection(pool):
    store, connections = pool
    with pytest.raises(pymysql.OperationalError):
        with store.connect() as db:
            db.connection.broken = True
    assert not connections[0].open
    with store.connect():
        pass
    assert len(connections) == 2


def test_shutdown_closes_idle_and_borrowed_connections(pool):
    store, connections = pool
    with store.connect():
        store.close()
        assert connections[0].open
    assert not connections[0].open
    with pytest.raises(ValueError, match="pool is closed"):
        with store.connect():
            pass
