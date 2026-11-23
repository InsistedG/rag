# Long-Term Memory Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add production-grade long-term conversation memory using MongoDB summaries and Milvus semantic memory snippets without breaking the existing RAG query flow.

**Architecture:** Keep the existing `chat_message` short-term history. Add `chat_memory_summary` in MongoDB for durable summaries and `chat_memory_collection` in Milvus for query-retrievable historical Q&A snippets. Integrate memory as an optional enhancement in `item_name_confirm` and `answer_output`, with all failures degraded to warnings.

**Tech Stack:** Python, FastAPI, LangGraph, MongoDB/PyMongo, Milvus/PyMilvus, BGE-M3 embeddings, LangChain ChatOpenAI.

---

### Task 1: Add Unit-Testable Long-Term Memory Formatting Helpers

**Files:**
- Create: `knowledge/tools/chat_memory_utils.py`
- Test: `knowledge/test/query/test_chat_memory_utils.py`

**Step 1: Write failing tests**

Create `knowledge/test/query/test_chat_memory_utils.py` with tests for:

```python
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
```

**Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest knowledge/test/query/test_chat_memory_utils.py -v
```

Expected: FAIL because `knowledge.tools.chat_memory_utils` does not exist.

**Step 3: Implement minimal helpers**

Create `knowledge/tools/chat_memory_utils.py` with:

- `build_memory_id(session_id, user_message_id, assistant_message_id) -> str`
- `truncate_text(text, max_chars) -> str`
- `format_memory_context(summary_doc, snippets, max_chars) -> str`

Do not add database code in this task.

**Step 4: Run test to verify it passes**

Run:

```bash
python -m pytest knowledge/test/query/test_chat_memory_utils.py -v
```

Expected: PASS.

---

### Task 2: Add MongoDB Summary Read/Write Utilities With Graceful Degradation

**Files:**
- Modify: `knowledge/tools/chat_memory_utils.py`
- Test: `knowledge/test/query/test_chat_memory_utils.py`

**Step 1: Write failing tests with monkeypatches**

Add tests that monkeypatch `_get_summary_collection()` with a fake collection:

```python
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
```

**Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest knowledge/test/query/test_chat_memory_utils.py -v
```

Expected: FAIL because Mongo helpers do not exist.

**Step 3: Implement Mongo helpers**

In `chat_memory_utils.py` add:

- `_get_summary_collection()`
- `get_memory_summary(session_id: str, user_id: str = "") -> dict`
- `upsert_memory_summary(...) -> None`

Use `knowledge.tools.mongo_history_utils._get_client()` and `get_config().mongo_db_name`.
Catch all exceptions and log warning.

**Step 4: Run test**

Run:

```bash
python -m pytest knowledge/test/query/test_chat_memory_utils.py -v
```

Expected: PASS.

---

### Task 3: Add Milvus Memory Snippet Read/Write Utilities

**Files:**
- Modify: `knowledge/tools/chat_memory_utils.py`
- Test: `knowledge/test/query/test_chat_memory_utils.py`

**Step 1: Write failing degradation tests**

Add tests:

```python
def test_retrieve_memory_snippets_returns_empty_when_disabled_client(monkeypatch):
    import knowledge.tools.chat_memory_utils as m

    monkeypatch.setattr(m, "get_milvus_client", lambda: None)
    assert m.retrieve_memory_snippets("s1", "query", top_k=3) == []


def test_save_memory_snippet_skips_empty_text():
    from knowledge.tools.chat_memory_utils import save_memory_snippet

    assert save_memory_snippet("s1", "", [], "", "") is False
```

**Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest knowledge/test/query/test_chat_memory_utils.py -v
```

Expected: FAIL because snippet functions do not exist.

**Step 3: Implement Milvus helpers**

In `chat_memory_utils.py` add:

- `_ensure_memory_collection(client, collection_name, vector_dim)`
- `save_memory_snippet(session_id, text, item_names, user_message_id, assistant_message_id, user_id="") -> bool`
- `retrieve_memory_snippets(session_id, query, top_k=3, user_id="") -> list`

Implementation notes:

- Use `generate_hybrid_embeddings`.
- Use `build_hybrid_search_requests` and `execute_hybrid_search`.
- Use filter `scope_type == "session" and scope_id == "{session_id}"`.
- Collection fields: `memory_id`, `scope_type`, `scope_id`, `user_id`, `text`, `item_names`, `created_at`, `dense_vector`, `sparse_vector`.
- Catch all exceptions and return `False` or `[]`.

**Step 4: Run test**

Run:

```bash
python -m pytest knowledge/test/query/test_chat_memory_utils.py -v
```

Expected: PASS.

---

### Task 4: Add Query Config and State Fields

**Files:**
- Modify: `knowledge/processor/query_process/config.py`
- Modify: `knowledge/processor/query_process/state.py`

**Step 1: Modify config**

Add fields to `QueryConfig`:

```python
long_term_memory_enabled: bool = field(
    default_factory=lambda: os.getenv("LONG_TERM_MEMORY_ENABLED", "true").lower() == "true"
)
memory_summary_every_n_messages: int = field(
    default_factory=lambda: int(os.getenv("MEMORY_SUMMARY_EVERY_N_MESSAGES", "6"))
)
memory_recent_limit: int = field(
    default_factory=lambda: int(os.getenv("MEMORY_RECENT_LIMIT", "10"))
)
memory_recall_top_k: int = field(
    default_factory=lambda: int(os.getenv("MEMORY_RECALL_TOP_K", "3"))
)
memory_summary_max_chars: int = field(
    default_factory=lambda: int(os.getenv("MEMORY_SUMMARY_MAX_CHARS", "1500"))
)
memory_snippet_max_chars: int = field(
    default_factory=lambda: int(os.getenv("MEMORY_SNIPPET_MAX_CHARS", "1200"))
)
memory_collection: str = field(
    default_factory=lambda: os.getenv("CHAT_MEMORY_COLLECTION", "chat_memory_collection")
)
```

**Step 2: Modify state**

Add TypedDict fields:

```python
memory_summary: str
memory_snippets: list
memory_profile: dict
```

Add defaults:

```python
"memory_summary": "",
"memory_snippets": [],
"memory_profile": {},
```

**Step 3: Verify import**

Run:

```bash
python -c "from knowledge.processor.query_process.config import get_config; from knowledge.processor.query_process.state import create_default_state; print(get_config().memory_recall_top_k, create_default_state()['memory_snippets'])"
```

Expected output includes:

```text
3 []
```

---

### Task 5: Enhance Prompts for Long-Term Memory

**Files:**
- Modify: `knowledge/processor/query_process/prompt.py`

**Step 1: Modify item-name extraction template**

Add placeholders:

```text
长期记忆：
{memory_context}
```

Keep existing JSON output contract unchanged.

**Step 2: Modify answer prompt**

Add placeholders before historical dialogue:

```text
【长期记忆】
{memory_context}
```

Keep reference-content-first instructions. Add one sentence:

```text
长期记忆只用于理解上下文和用户偏好，不得覆盖参考内容中的产品事实。
```

**Step 3: Verify formatting manually**

Run:

```bash
python -c "from knowledge.processor.query_process.prompt import ITEM_NAME_EXTRACT_TEMPLATE, ANSWER_PROMPT; print('memory_context' in ITEM_NAME_EXTRACT_TEMPLATE, 'memory_context' in ANSWER_PROMPT)"
```

Expected:

```text
True True
```

---

### Task 6: Read Long-Term Memory in ItemNameConfirmNode

**Files:**
- Modify: `knowledge/processor/query_process/nodes/item_name_confirm.py`

**Step 1: Add helper method**

Add `_load_long_term_memory(session_id, query)`:

- If `self.config.long_term_memory_enabled` is false, return empty summary/profile/snippets/context.
- Call `get_memory_summary(session_id)`.
- Call `retrieve_memory_snippets(session_id, query, top_k=self.config.memory_recall_top_k)`.
- Call `format_memory_context(...)`.
- Catch all exceptions and return empty values.

**Step 2: Update process**

After short history is loaded:

```python
memory = self._load_long_term_memory(session_id, query)
```

Pass `memory["context"]` into `_extract_item_names`.
Update state with:

```python
state["memory_summary"] = memory["summary"]
state["memory_profile"] = memory["profile"]
state["memory_snippets"] = memory["snippets"]
```

**Step 3: Update `_extract_item_names` signature**

Change:

```python
def _extract_item_names(self, query: str, history: List[Dict]) -> Dict[str, Any]:
```

to:

```python
def _extract_item_names(self, query: str, history: List[Dict], memory_context: str = "") -> Dict[str, Any]:
```

Format prompt with `memory_context=memory_context or "无长期记忆"`.

**Step 4: Smoke import**

Run:

```bash
python -c "from knowledge.processor.query_process.nodes.item_name_confirm import ItemNameConfirmNode; print(ItemNameConfirmNode().name)"
```

Expected:

```text
item_name_confirm
```

---

### Task 7: Inject and Update Long-Term Memory in AnswerOutputNode

**Files:**
- Modify: `knowledge/processor/query_process/nodes/answer_output.py`

**Step 1: Add memory formatting to prompt build**

In `_build_prompt`, call `format_memory_context` using state fields:

```python
memory_context = format_memory_context(
    {"summary": state.get("memory_summary", ""), "profile": state.get("memory_profile", {})},
    state.get("memory_snippets", []),
    self.config.memory_summary_max_chars + self.config.memory_snippet_max_chars,
)
```

Pass `memory_context=memory_context or "无长期记忆"` into `ANSWER_PROMPT.format(...)`.

**Step 2: Add long-term memory update after history write**

In `process`, after `_write_history(state)`, call:

```python
self._update_long_term_memory(state)
```

**Step 3: Implement `_update_long_term_memory`**

Behavior:

- If disabled, return.
- Read `session_id`, `original_query`, `answer`.
- If query or answer empty, return.
- Save current Q&A as memory snippet via `save_memory_snippet(...)`.
- Get recent messages with a wider limit such as `max(self.config.memory_recent_limit, self.config.memory_summary_every_n_messages * 2)`.
- If message count modulo summary interval is not 0, skip summary update.
- If interval reached, call LLM to summarize into JSON:
  - `summary`
  - `profile.preferred_style`
  - `profile.frequent_items`
  - `profile.confirmed_items`
  - `profile.known_context`
- Upsert summary using `upsert_memory_summary`.
- Catch all exceptions and log warning.

**Step 4: Smoke import**

Run:

```bash
python -c "from knowledge.processor.query_process.nodes.answer_output import AnswerOutputNode; print(AnswerOutputNode().name)"
```

Expected:

```text
answer_output
```

---

### Task 8: Add End-to-End Degradation Verification

**Files:**
- Test: `knowledge/test/query/test_long_term_memory_degradation.py`

**Step 1: Write degradation tests**

Create tests that monkeypatch memory helpers to raise exceptions and assert nodes still return normally.

Example:

```python
def test_item_name_confirm_memory_failure_does_not_crash(monkeypatch):
    import knowledge.processor.query_process.nodes.item_name_confirm as node_mod

    node = node_mod.ItemNameConfirmNode()
    monkeypatch.setattr(node.config, "long_term_memory_enabled", True)
    monkeypatch.setattr(node, "_get_history", lambda session_id: [])
    monkeypatch.setattr(node, "_save_message", lambda *args, **kwargs: "msg1")
    monkeypatch.setattr(node, "_extract_item_names", lambda *args, **kwargs: {"item_names": [], "rewritten_query": "q"})
    monkeypatch.setattr(node, "_match_and_align", lambda names: {})
    monkeypatch.setattr(node_mod, "get_memory_summary", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))

    state = node.process({"session_id": "s1", "original_query": "q", "item_names": [], "answer": ""})

    assert "answer" in state
```

Adjust imports to match the final implementation.

**Step 2: Run tests**

Run:

```bash
python -m pytest knowledge/test/query -v
```

Expected: PASS.

---

### Task 9: Manual Query Smoke Test

**Files:**
- No code changes.

**Step 1: Run import smoke checks**

Run:

```bash
python -c "from knowledge.processor.query_process.main_graph import query_app; print(query_app)"
```

Expected: prints a compiled graph object without import errors.

**Step 2: Run a query with long-term memory disabled**

Set `LONG_TERM_MEMORY_ENABLED=false` in environment or `.env`, then run a normal query through the existing app.

Expected:

- Query flow works as before.
- No memory collection is required.

**Step 3: Run a query with long-term memory enabled**

Set `LONG_TERM_MEMORY_ENABLED=true`.

Expected:

- If MongoDB/Milvus are available, memory summary/snippet attempts are logged.
- If any storage is unavailable, query still returns an answer.

---

### Task 10: Review and Cleanup

**Files:**
- All touched files.

**Step 1: Inspect diff**

Run:

```bash
git diff -- knowledge/tools/chat_memory_utils.py knowledge/processor/query_process/config.py knowledge/processor/query_process/state.py knowledge/processor/query_process/prompt.py knowledge/processor/query_process/nodes/item_name_confirm.py knowledge/processor/query_process/nodes/answer_output.py knowledge/test/query docs/plans
```

Expected:

- No unrelated formatting churn.
- No credentials in code.
- All long-term memory operations are optional and failure-tolerant.

**Step 2: Run focused tests**

Run:

```bash
python -m pytest knowledge/test/query -v
```

Expected: PASS.

**Step 3: Run graph import check**

Run:

```bash
python -c "from knowledge.processor.query_process.main_graph import query_app; print('ok')"
```

Expected:

```text
ok
```

