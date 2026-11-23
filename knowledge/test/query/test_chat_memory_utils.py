from knowledge.tools.chat_memory_utils import (
    build_memory_id,
    format_memory_context,
    truncate_text,
)


def test_build_memory_id_is_stable():
    first = build_memory_id("s1", "u1", "a1")
    second = build_memory_id("s1", "u1", "a1")
    assert first == second
    assert len(first) == 40


def test_truncate_text_keeps_short_text():
    assert truncate_text("abc", 10) == "abc"


def test_truncate_text_adds_suffix_for_long_text():
    assert truncate_text("abcdefghij", 6) == "abc..."


def test_format_memory_context_combines_summary_profile_and_snippets():
    summary = {
        "summary": "用户一直在排查显示器连接问题。",
        "profile": {
            "confirmed_items": ["华为擎云 L540"],
            "preferred_style": "步骤化",
        },
    }
    snippets = [
        {"text": "用户问 L540 如何连接显示器，助手回答使用 HDMI。"},
        {"text": "用户确认正在使用扩展坞。"},
    ]

    text = format_memory_context(summary, snippets, max_chars=500)

    assert "长期记忆摘要" in text
    assert "华为擎云 L540" in text
    assert "HDMI" in text


def test_get_memory_summary_returns_empty_on_error(monkeypatch):
    import knowledge.tools.chat_memory_utils as m

    def boom():
        raise RuntimeError("mongo down")

    monkeypatch.setattr(m, "_get_summary_collection", boom)

    assert m.get_memory_summary("s1") == {}


def test_upsert_memory_summary_writes_scope_fields(monkeypatch):
    import knowledge.tools.chat_memory_utils as m

    calls = {}

    class FakeCollection:
        def update_one(self, query, update, upsert=False):
            calls["query"] = query
            calls["update"] = update
            calls["upsert"] = upsert

    monkeypatch.setattr(m, "_get_summary_collection", lambda: FakeCollection())

    m.upsert_memory_summary(
        session_id="s1",
        summary="summary",
        profile={"confirmed_items": ["A"]},
        message_count=2,
        source_message_ids=["u1", "a1"],
    )

    assert calls["query"] == {"scope_type": "session", "scope_id": "s1"}
    assert calls["upsert"] is True
    assert calls["update"]["$set"]["summary"] == "summary"


def test_retrieve_memory_snippets_returns_empty_when_disabled_client(monkeypatch):
    import knowledge.tools.chat_memory_utils as m

    monkeypatch.setattr(m, "get_milvus_client", lambda: None)

    assert m.retrieve_memory_snippets("s1", "query", top_k=3) == []


def test_save_memory_snippet_skips_empty_text():
    from knowledge.tools.chat_memory_utils import save_memory_snippet

    assert save_memory_snippet("s1", "", [], "", "") is False
