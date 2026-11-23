"""RAG 检索评估与优化模块（基于 RAGAS）

自动修复 ragas 导入兼容性问题（langchain-community 不再包含 ChatVertexAI）。

核心组件：
  - TestsetGenerator: 从知识库自动生成测试集（含 reference answer）
  - RetrievalEvaluator: 基于 RAGAS 的评估（Faithfulness/AnswerRelevancy/ContextPrecision/ContextRecall）
  - RecallOptimizer: 超参数搜索优化

使用示例:
    python -m knowledge.evaluation.testset_generator
    python -m knowledge.evaluation.retrieval_evaluator -t data/testset_v1.json
    python -m knowledge.evaluation.recall_optimizer -t data/testset_v1.json
"""

import sys
import types


def _patch_ragas_vertexai_import():
    """修复 ragas 对 langchain_community.chat_models.vertexai 的硬导入。

    ChatVertexAI 在 langchain-community >= 0.3 中被移除，
    移入 langchain-google-vertexai 独立包。Ragas 0.2.x ~ 0.4.x 仍尝试
    从旧路径导入。提供一个空壳模块满足导入，不影响实际使用。
    """
    module_name = "langchain_community.chat_models.vertexai"
    if module_name in sys.modules:
        return

    from langchain_community import chat_models

    fake = types.ModuleType(module_name)
    fake.ChatVertexAI = type("ChatVertexAI", (), {})  # type: ignore[assignment]
    sys.modules[module_name] = fake
    chat_models.vertexai = fake  # type: ignore[attr-defined]


_patch_ragas_vertexai_import()
