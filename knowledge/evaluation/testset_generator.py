"""测试集生成器

从知识库随机采样切片，使用 LLM 为每条切片生成自然语言查询和参考答案，
记录切片 chunk_id 作为 ground truth，输出 JSON 测试集供 RAGAS 评估使用。
"""

import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from knowledge.evaluation.config import get_eval_config

logger = logging.getLogger("eval.generator")

# 单跳：从一个 chunk 生成 query
QUERY_GEN_PROMPT = """你是一个 RAG 系统的测试数据标注员。根据以下文档片段，完成两项任务：

1. 生成 1 个自然语言问题，该问题的答案可以从文档中找到
2. 根据文档内容写出该问题的参考答案

要求：
- **问题中必须包含商品名称 "{item_name}"**，模拟用户询问该产品的场景
- 问题要具体、有明确指向
- 答案必须完全基于文档内容，不要编造
- 使用中文
- 输出格式为 JSON：{{"query": "问题内容", "reference": "参考答案"}}

文档标题：{title}
文档内容：
{content}

直接输出 JSON："""

# 多跳：从两个 chunk 生成需要跨片段拼接答案的 query
MULTI_HOP_PROMPT = """你是一个 RAG 系统的测试数据标注员。以下是从产品 **{item_name}** 文档中提取的两个不连续片段。
请生成 1 个**需要同时用到两个片段的信息才能完整回答**的问题，并写出完整参考答案。

要求：
- 问题中必须包含商品名称 "{item_name}"
- 答案**必须同时依赖片段A和片段B的信息**，仅靠一个片段无法完整回答
- 问题要自然，模拟真实用户的多步咨询场景
- 使用中文
- 输出格式为 JSON：{{"query": "问题内容", "reference": "参考答案（综合两个片段）"}}

【片段 A】
{content_a}

【片段 B】
{content_b}

直接输出 JSON："""


class TestsetGenerator:
    """基于 LLM 从知识库自动生成 RAGAS 兼容测试集。"""

    def __init__(self, config=None):
        self.config = config or get_eval_config()

    def generate(self, output_path: str = None, mode: str = "single") -> List[Dict[str, Any]]:
        """生成测试集并保存。

        Args:
            output_path: 输出文件路径。
            mode: "single" 单跳, "multi" 多跳, "mixed" 混合（各半）。

        Returns:
            测试集记录列表。
        """
        logger.info(f"步骤1: 从 Milvus 采样切片 (mode={mode})")
        chunks = self._sample_chunks()
        logger.info(f"采样到 {len(chunks)} 条切片")

        if not chunks:
            logger.warning("未采样到切片，请检查 Milvus 中是否有数据")
            return []

        items = []
        target = self.config.eval_testset_size

        if mode == "multi":
            logger.info("步骤2: LLM 生成多跳 query + reference")
            items = self._generate_multi_hop(chunks, target)
        elif mode == "mixed":
            half = max(1, target // 2)
            logger.info(f"步骤2a: 生成单跳 ({half} 条)")
            single = self._generate_qa_pairs(chunks[:half])
            logger.info(f"步骤2b: 生成多跳 ({half} 条)")
            multi = self._generate_multi_hop(chunks, half)
            items = single + multi
        else:
            logger.info("步骤2: LLM 生成单跳 query + reference")
            items = self._generate_qa_pairs(chunks)

        logger.info(f"生成 {len(items)} 条测试用例")

        if output_path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            data_dir = Path(self.config.eval_data_dir)
            data_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(data_dir / f"testset_{mode}_{ts}.json")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        logger.info(f"测试集已保存: {output_path} ({len(items)} 条)")

        return items

    # ------------------------------------------------------------------ #
    #  采样                                                                #
    # ------------------------------------------------------------------ #

    def _sample_chunks(self) -> List[Dict[str, Any]]:
        """从 Milvus 随机采样切片。"""
        try:
            from knowledge.tools.milvus_utils import get_milvus_client
        except ImportError as e:
            logger.error(f"无法导入 Milvus 客户端: {e}")
            return []

        client = get_milvus_client()
        collection = self.config.eval_sample_collection
        sample_limit = self.config.eval_sample_limit

        try:
            stats = client.get_collection_stats(collection)
            total = stats.get("row_count", 0)
        except Exception:
            total = 0

        if total == 0:
            logger.warning(f"集合 {collection} 为空")
            return []

        offset = (
            random.randint(0, max(0, total - sample_limit - 1))
            if total > sample_limit else 0
        )

        try:
            results = client.query(
                collection_name=collection,
                filter="chunk_id >= 0",
                output_fields=["chunk_id", "content", "title", "item_name"],
                limit=sample_limit,
                offset=offset,
            )
        except Exception as e:
            logger.error(f"Milvus 查询失败: {e}")
            return []

        return [r for r in results if r.get("content", "").strip()]

    # ------------------------------------------------------------------ #
    #  LLM 生成                                                            #
    # ------------------------------------------------------------------ #

    def _generate_qa_pairs(self, chunks: List[Dict]) -> List[Dict[str, Any]]:
        """对每条切片调用 LLM 生成 query + reference answer。"""
        try:
            from knowledge.tools.llm_utils import get_llm_client
        except ImportError as e:
            logger.error(f"无法导入 LLM 客户端: {e}")
            return []

        llm = get_llm_client(model=self.config.eval_llm_model, temperature=0.7)
        items = []
        target = min(self.config.eval_testset_size, len(chunks))
        selected = random.sample(chunks, target) if target < len(chunks) else chunks

        for i, chunk in enumerate(selected):
            try:
                prompt = QUERY_GEN_PROMPT.format(
                    item_name=chunk.get("item_name", "该产品"),
                    title=chunk.get("title", ""),
                    content=chunk.get("content", ""),
                )
                response = llm.invoke(prompt)
                raw = response.content.strip()

                # 提取 JSON（有些 LLM 会在 JSON 外加 markdown 代码块）
                if "```" in raw:
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                qa = json.loads(raw)
                query = qa.get("query", "").strip()
                reference = qa.get("reference", "").strip()

                if query:
                    items.append({
                        "query_id": f"q{i + 1:04d}",
                        "query": query,
                        "reference": reference,
                        "item_names": (
                            [chunk.get("item_name", "")]
                            if chunk.get("item_name") else []
                        ),
                        "expected_chunk_ids": [chunk["chunk_id"]],
                        "source": "llm",
                        "source_chunk_title": chunk.get("title", ""),
                    })
                    logger.debug(f"[{i + 1}/{target}] {query[:60]}...")
                else:
                    logger.warning(f"[{i + 1}/{target}] LLM 返回空 query")

            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"[{i + 1}/{target}] 生成失败: {e}")
                continue

        return items


    def _generate_multi_hop(self, chunks: List[Dict], target: int) -> List[Dict[str, Any]]:
        """从同商品 chunk 对生成多跳测试查询。"""
        try:
            from knowledge.tools.llm_utils import get_llm_client
        except ImportError:
            return []

        # 按 item_name 分组，只保留有多个 chunk 的商品
        from collections import defaultdict
        groups = defaultdict(list)
        for c in chunks:
            name = c.get("item_name", "")
            if name:
                groups[name].append(c)

        # 随机配对
        import random as _random
        pairs = []
        for name, group in groups.items():
            if len(group) >= 2:
                a, b = _random.sample(group, 2)
                pairs.append((name, a, b))
        _random.shuffle(pairs)

        llm = get_llm_client(model=self.config.eval_llm_model, temperature=0.7)
        items = []
        for i, (iname, ca, cb) in enumerate(pairs[:target]):
            try:
                prompt = MULTI_HOP_PROMPT.format(
                    item_name=iname,
                    content_a=ca.get("content", ""),
                    content_b=cb.get("content", ""),
                )
                resp = llm.invoke(prompt)
                raw = resp.content.strip()
                if "```" in raw:
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                qa = json.loads(raw)
                query = qa.get("query", "").strip()
                reference = qa.get("reference", "").strip()
                if query:
                    items.append({
                        "query_id": f"m{i + 1:04d}",
                        "query": query,
                        "reference": reference,
                        "item_names": [iname] if iname else [],
                        "expected_chunk_ids": [ca["chunk_id"], cb["chunk_id"]],
                        "source": "llm-multi-hop",
                    })
                    logger.debug(f"[{i + 1}/{target}] multi-hop: {query[:60]}...")
            except Exception as e:
                logger.warning(f"[{i + 1}/{target}] 多跳生成失败: {e}")
        return items


# ================================================================== #
#  测试入口                                                              #
# ================================================================== #

if __name__ == "__main__":
    import argparse
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    p = argparse.ArgumentParser(description="RAGAS 测试集生成器")
    p.add_argument("--mode", "-m", default="mixed",
                   choices=["single", "multi", "mixed"],
                   help="single=单跳, multi=多跳, mixed=各半 (默认)")
    p.add_argument("--output", "-o", default=None, help="输出路径")
    args = p.parse_args()

    generator = TestsetGenerator()
    result = generator.generate(output_path=args.output, mode=args.mode)
    print(f"\n生成测试集 ({args.mode}): {len(result)} 条")
    for item in result[:3]:
        print(f"  [{item['query_id']}] Q: {item['query'][:80]}...")
        print(f"       chunks={item['expected_chunk_ids']}")
    if len(result) > 3:
        print(f"  ... 共 {len(result)} 条")
