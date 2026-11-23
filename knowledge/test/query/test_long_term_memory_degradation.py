def test_item_name_confirm_memory_failure_does_not_crash(monkeypatch):
    import knowledge.processor.query_process.nodes.item_name_confirm as node_mod
    import knowledge.tools.chat_memory_utils as memory_utils

    node = node_mod.ItemNameConfirmNode()
    monkeypatch.setattr(node.config, "long_term_memory_enabled", True)
    monkeypatch.setattr(node, "_get_history", lambda session_id: [])
    monkeypatch.setattr(node, "_save_message", lambda *args, **kwargs: "msg1")
    monkeypatch.setattr(
        node,
        "_extract_item_names",
        lambda *args, **kwargs: {"item_names": [], "rewritten_query": "q"},
    )
    monkeypatch.setattr(node, "_match_and_align", lambda names: {})
    monkeypatch.setattr(
        memory_utils,
        "get_memory_summary",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    state = node.process({
        "session_id": "s1",
        "original_query": "q",
        "item_names": [],
        "answer": "",
    })

    assert "answer" in state
    assert state.get("memory_summary", "") == ""


def test_answer_output_memory_update_failure_does_not_crash(monkeypatch):
    node_mod = __import__(
        "knowledge.processor.query_process.nodes.answer_output",
        fromlist=["AnswerOutputNode"],
    )
    import knowledge.tools.chat_memory_utils as memory_utils

    node = node_mod.AnswerOutputNode()
    monkeypatch.setattr(node.config, "long_term_memory_enabled", True)
    monkeypatch.setattr(node, "_invoke_generate", lambda prompt: "答案")
    monkeypatch.setattr(node, "_write_history", lambda state: None)
    monkeypatch.setattr(
        memory_utils,
        "save_memory_snippet",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    state = {
        "session_id": "s1",
        "original_query": "q",
        "rewritten_query": "q",
        "item_names": [],
        "reranked_docs": [],
        "history": [],
        "kg_triples": [],
        "memory_summary": "",
        "memory_profile": {},
        "memory_snippets": [],
        "is_stream": False,
    }

    result = node.process(state)
    assert result["answer"] == "答案"
