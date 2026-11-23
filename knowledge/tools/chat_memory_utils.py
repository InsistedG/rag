"""长期对话记忆工具。

该模块封装长期记忆的格式化、摘要存取和语义片段存取。
所有外部存储操作都应失败降级，不能阻断主查询链路。
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List

from knowledge.tools.milvus_utils import (
    build_hybrid_search_requests,
    execute_hybrid_search,
    get_milvus_client,
)

logger = logging.getLogger("query.chat_memory")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def build_memory_id(
    session_id: str,
    user_message_id: str,
    assistant_message_id: str,
) -> str:
    """生成稳定的长期记忆片段 ID。"""
    raw = f"{session_id}:{user_message_id}:{assistant_message_id}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def truncate_text(text: str, max_chars: int) -> str:
    """按字符数截断文本，预留省略号。"""
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return "." * max_chars
    return text[: max_chars - 3] + "..."


def format_memory_context(
    summary_doc: Dict[str, Any] | None,
    snippets: List[Dict[str, Any]] | None,
    max_chars: int,
) -> str:
    """将长期摘要和召回片段格式化为 prompt 上下文。"""
    summary_doc = summary_doc or {}
    snippets = snippets or []

    parts: List[str] = []

    summary = str(summary_doc.get("summary") or "").strip()
    profile = summary_doc.get("profile") or {}
    if summary or profile:
        lines = ["【长期记忆摘要】"]
        if summary:
            lines.append(summary)
        if profile:
            profile_text = json.dumps(profile, ensure_ascii=False)
            lines.append(f"用户/会话画像：{profile_text}")
        parts.append("\n".join(lines))

    snippet_lines = []
    for idx, snippet in enumerate(snippets, 1):
        text = str(snippet.get("text") or "").strip()
        if text:
            snippet_lines.append(f"[记忆{idx}] {text}")

    if snippet_lines:
        parts.append("【相关历史记忆】\n" + "\n".join(snippet_lines))

    return truncate_text("\n\n".join(parts), max_chars)


def _get_summary_collection():
    """获取长期摘要集合。"""
    from knowledge.processor.query_process.config import get_config
    from knowledge.tools.mongo_history_utils import _get_client

    config = get_config()
    return _get_client()[config.mongo_db_name]["chat_memory_summary"]


def get_memory_summary(
    session_id: str,
    user_id: str = "",
) -> Dict[str, Any]:
    """读取当前 session 的长期摘要，失败时返回空字典。"""
    if not session_id:
        return {}

    try:
        collection = _get_summary_collection()
        doc = collection.find_one({"scope_type": "session", "scope_id": session_id})
        if not doc:
            return {}
        doc["_id"] = str(doc["_id"])
        return doc
    except Exception as e:
        logger.warning(f"读取长期记忆摘要失败: {e}")
        return {}


def upsert_memory_summary(
    session_id: str,
    summary: str,
    profile: Dict[str, Any] | None = None,
    message_count: int = 0,
    source_message_ids: List[str] | None = None,
    user_id: str = "",
) -> None:
    """写入或更新当前 session 的长期摘要，失败时降级为 warning。"""
    if not session_id:
        return

    try:
        collection = _get_summary_collection()
        now = _utc_now()
        collection.update_one(
            {"scope_type": "session", "scope_id": session_id},
            {
                "$set": {
                    "scope_type": "session",
                    "scope_id": session_id,
                    "user_id": user_id,
                    "summary": summary or "",
                    "profile": profile or {},
                    "message_count": message_count,
                    "source_message_ids": source_message_ids or [],
                    "updated_at": now,
                    "version": 1,
                }
            },
            upsert=True,
        )
    except Exception as e:
        logger.warning(f"写入长期记忆摘要失败: {e}")


def _escape_expr_value(value: str) -> str:
    """转义 Milvus 表达式中的字符串值。"""
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _get_memory_collection_name() -> str:
    from knowledge.processor.query_process.config import get_config

    config = get_config()
    return getattr(config, "memory_collection", "chat_memory_collection")


def _ensure_memory_collection(client, collection_name: str, vector_dim: int) -> None:
    """确保长期记忆 Milvus 集合存在。"""
    if client.has_collection(collection_name=collection_name):
        return

    from pymilvus import DataType

    schema = client.create_schema(enable_dynamic_fields=True)
    schema.add_field(
        field_name="memory_id",
        datatype=DataType.VARCHAR,
        is_primary=True,
        auto_id=False,
        max_length=100,
    )
    schema.add_field(field_name="scope_type", datatype=DataType.VARCHAR, max_length=32)
    schema.add_field(field_name="scope_id", datatype=DataType.VARCHAR, max_length=128)
    schema.add_field(field_name="user_id", datatype=DataType.VARCHAR, max_length=128)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="item_names", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="created_at", datatype=DataType.VARCHAR, max_length=64)
    schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field(
        field_name="dense_vector",
        datatype=DataType.FLOAT_VECTOR,
        dim=vector_dim,
    )

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="dense_vector",
        index_name="dense_vector_index",
        index_type="AUTOINDEX",
        metric_type="IP",
    )
    index_params.add_index(
        field_name="sparse_vector",
        index_name="sparse_inverted_index",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="IP",
        params={"inverted_index_algo": "DAAT_MAXSCORE"},
    )

    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
    )


def save_memory_snippet(
    session_id: str,
    text: str,
    item_names: List[str] | None,
    user_message_id: str,
    assistant_message_id: str,
    user_id: str = "",
) -> bool:
    """保存一条长期记忆问答片段到 Milvus，失败时返回 False。"""
    text = (text or "").strip()
    if not session_id or not text:
        return False

    try:
        client = get_milvus_client()
        if client is None:
            return False

        from knowledge.tools.embedding_utils import generate_hybrid_embeddings

        collection_name = _get_memory_collection_name()
        embeddings = generate_hybrid_embeddings([text])
        dense_vector = embeddings["dense"][0]
        sparse_vector = embeddings["sparse"][0]
        vector_dim = len(dense_vector)

        _ensure_memory_collection(client, collection_name, vector_dim)

        memory_id = build_memory_id(
            session_id,
            user_message_id or text,
            assistant_message_id or "",
        )
        existing = client.query(
            collection_name=collection_name,
            filter=f'memory_id == "{_escape_expr_value(memory_id)}"',
            output_fields=["memory_id"],
            limit=1,
        )
        if existing:
            return True

        data = {
            "memory_id": memory_id,
            "scope_type": "session",
            "scope_id": session_id,
            "user_id": user_id,
            "text": text,
            "item_names": json.dumps(item_names or [], ensure_ascii=False),
            "created_at": _utc_now().isoformat(),
            "dense_vector": dense_vector,
            "sparse_vector": sparse_vector,
        }
        client.insert(collection_name=collection_name, data=[data])
        return True
    except Exception as e:
        logger.warning(f"写入长期记忆片段失败: {e}")
        return False


def retrieve_memory_snippets(
    session_id: str,
    query: str,
    top_k: int = 3,
    user_id: str = "",
) -> List[Dict[str, Any]]:
    """按当前 query 召回长期记忆片段，失败时返回空列表。"""
    query = (query or "").strip()
    if not session_id or not query or top_k <= 0:
        return []

    try:
        client = get_milvus_client()
        if client is None:
            return []

        from knowledge.tools.embedding_utils import generate_hybrid_embeddings

        collection_name = _get_memory_collection_name()
        if not client.has_collection(collection_name=collection_name):
            return []

        embeddings = generate_hybrid_embeddings([query])
        session_expr = (
            f'scope_type == "session" and '
            f'scope_id == "{_escape_expr_value(session_id)}"'
        )
        if user_id:
            session_expr += f' and user_id == "{_escape_expr_value(user_id)}"'

        requests = build_hybrid_search_requests(
            dense_vector=embeddings["dense"][0],
            sparse_vector=embeddings["sparse"][0],
            filter_expr=session_expr,
            top_k=top_k,
        )
        results = execute_hybrid_search(
            client=client,
            collection_name=collection_name,
            search_requests=requests,
            ranker_weights=(0.5, 0.5),
            top_k=top_k,
            output_fields=["memory_id", "text", "item_names", "created_at"],
        )

        hits = results[0] if results else []
        snippets: List[Dict[str, Any]] = []
        for hit in hits:
            entity = hit.get("entity", hit)
            item_names = entity.get("item_names", "[]")
            if isinstance(item_names, str):
                try:
                    item_names = json.loads(item_names)
                except Exception:
                    item_names = []
            snippets.append({
                "memory_id": entity.get("memory_id"),
                "text": entity.get("text", ""),
                "item_names": item_names,
                "created_at": entity.get("created_at", ""),
                "score": hit.get("distance"),
            })
        return snippets
    except Exception as e:
        logger.warning(f"召回长期记忆片段失败: {e}")
        return []
