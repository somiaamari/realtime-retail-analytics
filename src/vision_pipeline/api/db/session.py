"""Database engine and session lifecycle helpers."""

from collections.abc import Iterator

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool


def create_database_engine(database_url: str) -> Engine:
    """Create a SQLite/PostgreSQL engine with safe SQLite connection defaults."""
    options: dict[str, object] = {"pool_pre_ping": True}
    if database_url in ("sqlite://", "sqlite:///:memory:"):
        options.update(
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
    elif database_url.startswith("sqlite:"):
        options["connect_args"] = {"check_same_thread": False}
    engine = create_engine(database_url, **options)
    if database_url.startswith("sqlite:"):
        @event.listens_for(engine, "connect")
        def set_sqlite_pragmas(connection: object, _record: object) -> None:
            cursor = connection.cursor()  # type: ignore[attr-defined]
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.close()
    return engine


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Create a typed, non-expiring session factory."""
    return sessionmaker(bind=engine, expire_on_commit=False, class_=Session)


def get_db_session(factory: sessionmaker[Session]) -> Iterator[Session]:
    """Yield and close a database session."""
    with factory() as session:
        yield session
