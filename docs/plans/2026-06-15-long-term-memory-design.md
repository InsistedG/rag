# 长期记忆设计方案

## 背景

当前项目已经具备短期对话记忆：查询链路会从 MongoDB 的 `chat_message` 中读取当前 `session_id` 最近若干条消息，用于商品名确认、指代消解和问题改写。这个机制适合短期上下文，但无法稳定利用早期对话中的商品确认、用户偏好和阶段性问题结论。

长期记忆的目标不是把更多历史直接塞进 prompt，而是将早期对话沉淀为可控、可检索、可降级的记忆资产。

## 目标

1. 保留当前短期记忆能力，避免破坏现有查询链路。
2. 增加长期摘要记忆，用于沉淀会话阶段结论、已确认商品和用户偏好。
3. 增加语义片段记忆，用于按当前问题召回早期相关问答。
4. 预留 `user_id`，当前先按 `session_id` 生效，未来支持跨会话用户记忆。
5. 长期记忆失败不能阻断主问答链路。

## 总体架构

长期记忆采用三层结构：

```text
短期记忆：MongoDB chat_message 最近 N 条
长期摘要：MongoDB chat_memory_summary
长期片段：Milvus chat_memory_collection
```

查询时组合：

```text
short_history
+ memory_summary
+ retrieved_memory_snippets
+ RAG docs
+ KG triples
-> final prompt
```

新增工具模块：

```text
knowledge/tools/chat_memory_utils.py
```

该模块统一封装长期记忆的 MongoDB 和 Milvus 读写，避免存储细节散落在各个 LangGraph 节点中。

## 数据模型

### MongoDB: chat_memory_summary

```json
{
  "scope_type": "session",
  "scope_id": "session_id",
  "user_id": "",
  "summary": "本会话长期摘要",
  "profile": {
    "preferred_style": "步骤化、简洁",
    "frequent_items": ["华为擎云 L540"],
    "confirmed_items": ["华为擎云 L540"],
    "known_context": ["用户正在排查外接显示器问题"]
  },
  "message_count": 24,
  "source_message_ids": ["..."],
  "updated_at": "...",
  "version": 1
}
```

`scope_type` 第一版固定为 `session`，未来支持 `user`。`scope_id` 第一版使用 `session_id`。`user_id` 可为空。

### Milvus: chat_memory_collection

```json
{
  "memory_id": "hash(session_id + user_message_id + assistant_message_id)",
  "scope_type": "session",
  "scope_id": "session_id",
  "user_id": "",
  "text": "用户问... 助手答...",
  "item_names": ["..."],
  "dense_vector": [...],
  "sparse_vector": {...},
  "created_at": "..."
}
```

`memory_id` 用于去重，避免同一轮问答重复写入。

## 配置项

在 `QueryConfig` 中增加：

```python
long_term_memory_enabled: bool
memory_summary_every_n_messages: int
memory_recent_limit: int
memory_recall_top_k: int
memory_summary_max_chars: int
memory_snippet_max_chars: int
memory_collection: str
```

建议默认值：

```text
LONG_TERM_MEMORY_ENABLED=true
MEMORY_SUMMARY_EVERY_N_MESSAGES=6
MEMORY_RECENT_LIMIT=10
MEMORY_RECALL_TOP_K=3
MEMORY_SUMMARY_MAX_CHARS=1500
MEMORY_SNIPPET_MAX_CHARS=1200
CHAT_MEMORY_COLLECTION=chat_memory_collection
```

## QueryGraphState 扩展

增加字段：

```python
memory_summary: str
memory_snippets: list
memory_profile: dict
```

这些字段只作为查询增强信息，不作为主链路必需字段。

## 读路径

在 `item_name_confirm` 开始时：

1. 读取最近短期历史。
2. 读取 `chat_memory_summary`。
3. 用当前 query 从 `chat_memory_collection` 召回 topK 历史片段。
4. 将长期摘要和召回片段传入商品名提取 prompt。
5. 将 `memory_summary`、`memory_profile`、`memory_snippets` 写入 state。

在 `answer_output` 构建 prompt 时，加入：

```text
【长期记忆摘要】
...

【相关历史记忆】
...
```

prompt 事实优先级为：RAG 检索结果高于长期记忆。长期记忆用于上下文连续性和偏好，不覆盖产品文档事实。

## 写路径

每轮继续写入 `chat_message`。

在 `answer_output` 生成并保存助手回答后，维护长期记忆：

1. 根据当前 session 的消息数判断是否达到摘要周期。
2. 达到周期时，调用 LLM 更新 `chat_memory_summary`。
3. 将当前问答 pair 组装为 memory snippet。
4. 生成 BGE-M3 dense/sparse 向量。
5. 写入 Milvus `chat_memory_collection`。

第一版同步执行，但所有错误降级为 warning，不影响 `/query` 返回。后续如性能压力明显，再迁移到后台任务或队列。

## 降级策略

长期记忆是增强能力，不是主链路硬依赖：

```text
MongoDB summary 读取失败 -> summary 为空，继续问答
Milvus memory 召回失败 -> snippets 为空，继续问答
LLM summary 更新失败 -> 只写原始历史，继续问答
Embedding 写入失败 -> 跳过向量记忆，继续问答
Milvus collection 创建失败 -> 关闭本轮片段写入，继续问答
```

所有异常只记录 warning。

## 性能策略

1. `memory_recall_top_k` 默认 3。
2. `memory_summary_max_chars` 限制摘要长度。
3. `memory_snippet_max_chars` 限制召回片段总长度。
4. prompt 预算优先级：RAG docs、KG triples、短期历史、长期摘要、相关历史片段。
5. Milvus 写入使用 `memory_id` 去重。
6. Milvus collection 不存在时自动创建。
7. BGE-M3 仍复用现有单例模型，避免重复加载。

## 第一版实施范围

1. 新增 `knowledge/tools/chat_memory_utils.py`。
2. 新增长期记忆配置项。
3. 扩展 `QueryGraphState`。
4. 增强 `ITEM_NAME_EXTRACT_TEMPLATE` 和 `ANSWER_PROMPT`。
5. 在 `item_name_confirm` 读取长期记忆。
6. 在 `answer_output` 注入长期记忆并更新长期记忆。
7. 增加基础测试。
