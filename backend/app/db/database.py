"""数据库连接与会话管理"""
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings
from loguru import logger

settings = get_settings()

engine = create_engine(
    settings.mysql_url,
    pool_pre_ping=True,
    pool_recycle=1800,
    pool_size=10,
    max_overflow=20,
    echo=False,
    future=True,
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    """ORM 基类"""


def get_db() -> Generator[Session, None, None]:
    """FastAPI 依赖：请求级数据库会话"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """脚本/后台任务使用的会话上下文"""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_db() -> None:
    """建表（若不存在）"""
    # 确保数据库存在
    _ensure_database()
    # 导入模型以注册元数据
    from app.models import conversation, doc_vector, document, qa_cache  # noqa: F401

    Base.metadata.create_all(bind=engine)
    _ensure_columns()
    logger.info("MySQL 数据表初始化完成")


def _ensure_columns() -> None:
    """
    给已存在的表补新增的列

    create_all 只会"建不存在的表"，不会给老表加列。开发过程中表结构改过好几次，
    这里对新增的列做一次幂等补齐，避免每次都要手动跑迁移脚本。
    """
    wanted = {
        "qa_cache": {
            "vector": "LONGBLOB NULL COMMENT '问题向量 float32（L2 归一化）'",
            "vector_model": "VARCHAR(128) NULL COMMENT '生成向量的模型标识'",
        },
    }
    for table, columns in wanted.items():
        try:
            with engine.connect() as conn:
                existing = {row[0] for row in conn.execute(text(f"SHOW COLUMNS FROM `{table}`"))}
                added = []
                for name, ddl in columns.items():
                    if name in existing:
                        continue
                    conn.execute(text(f"ALTER TABLE `{table}` ADD COLUMN `{name}` {ddl}"))
                    added.append(name)
                conn.commit()
            if added:
                logger.info(f"{table} 补充列：{added}")
        except Exception as exc:  # 表还没建/权限不足都不该挡住启动
            logger.warning(f"{table} 补列失败（可忽略）: {exc}")


def _ensure_database() -> None:
    """若目标库不存在则创建"""
    server_url = (
        f"mysql+pymysql://{settings.mysql_user}:{settings.mysql_password}"
        f"@{settings.mysql_host}:{settings.mysql_port}/?charset=utf8mb4"
    )
    server_engine = create_engine(server_url, future=True)
    try:
        with server_engine.connect() as conn:
            conn.execute(
                text(
                    f"CREATE DATABASE IF NOT EXISTS `{settings.mysql_database}` "
                    "DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
                )
            )
            conn.commit()
    finally:
        server_engine.dispose()


def check_mysql() -> bool:
    """数据库连通性检查"""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        logger.error(f"MySQL 连接失败: {exc}")
        return False
