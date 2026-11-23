"""纯单跳测试集生成器

从知识库随机采样切片，LLM 为每条切片生成 1 个直搜/单跳 query + reference answer。
每条 query 的 ground truth 是生成它的那个 chunk。

与 testset_generator.py 的区别：仅生成单跳，不含多跳。
"""

import json
import logging
import random
from datetime import datetime
from pathlib import Path

from knowledge.evaluation.config import get_eval_config

logger = logging.getLogger("eval.single_gen")

QUERY_GEN_PROMPT = """你是一个 RAG 系统的测试数据标注员。根据以下文档片段，生成测试数据。

要求：
- 生成 1 个自然语言问题，模拟用户查询该产品的场景
- 问题中必须包含商品名称 "{item_name}"
- 问题类型：直接搜索/事实型（what/是否/如何），答案能从文档中找到
- 写出该问题的参考答案，完全基于文档内容
- 使用中文
- 输出 JSON：{{"query": "问题内容", "reference": "参考答案"}}

文档标题：{title}
文档内容：
{content}

直接输出 JSON："""


class SingleHopGenerator:
    """纯单跳测试集生成器。"""

    def __init__(self, config=None):
        self.config = config or get_eval_config()

    def generate(self, output_path: str = None) -> list:
        logger.info("步骤1: 从 Milvus 采样切片")
        chunks = self._sample_chunks()
        logger.info(f"采样到 {len(chunks)} 条切片")

        if not chunks:
            logger.warning("未采样到切片")
            return []

        logger.info("步骤2: LLM 生成单跳 query + reference")
        items = self._generate_single_hop(chunks)
        logger.info(f"生成 {len(items)} 条")

        if output_path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            data_dir = Path(self.config.eval_data_dir)
            data_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(data_dir / f"testset_single_{ts}.json")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        logger.info(f"测试集已保存: {output_path} ({len(items)} 条)")
        return items

    def _sample_chunks(self) -> list:
        try:
            from knowledge.tools.milvus_utils import get_milvus_client
        except ImportError as e:
            logger.error(f"无法导入 Milvus: {e}")
            return []

        client = get_milvus_client()
        collection = self.config.eval_sample_collection
        limit = self.config.eval_sample_limit

        try:
            total = client.get_collection_stats(collection).get("row_count", 0)
        except Exception:
            total = 0
        if total == 0:
            return []

        offset = random.randint(0, max(0, total - limit - 1)) if total > limit else 0
        try:
            results = client.query(
                collection_name=collection, filter="chunk_id >= 0",
                output_fields=["chunk_id", "content", "title", "item_name"],
                limit=limit, offset=offset,
            )
        except Exception as e:
            logger.error(f"Milvus 查询失败: {e}")
            return []

        return [r for r in results if r.get("content", "").strip()]

    def _generate_single_hop(self, chunks: list) -> list:
        try:
            from knowledge.tools.llm_utils import get_llm_client
        except ImportError:
            return []

        llm = get_llm_client(model=self.config.eval_llm_model, temperature=0.7)
        target = min(self.config.eval_testset_size, len(chunks))
        selected = random.sample(chunks, target) if target < len(chunks) else chunks
        items = []

        for i, chunk in enumerate(selected):
            try:
                prompt = QUERY_GEN_PROMPT.format(
                    item_name=chunk.get("item_name", "该产品"),
                    title=chunk.get("title", ""),
                    content=chunk.get("content", ""),
                )
                raw = llm.invoke(prompt).content.strip()
                if "```" in raw:
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                qa = json.loads(raw)
                query = qa.get("query", "").strip()
                reference = qa.get("reference", "").strip()
                if query:
                    items.append({
                        "query_id": f"s{i + 1:04d}",
                        "query": query,
                        "reference": reference,
                        "item_names": [chunk.get("item_name", "")] if chunk.get("item_name") else [],
                        "expected_chunk_ids": [chunk["chunk_id"]],
                        "source": "llm-single-hop",
                    })
            except Exception as e:
                logger.warning(f"[{i + 1}/{target}] 生成失败: {e}")
        return items


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="纯单跳测试集生成器")
    p.add_argument("--output", "-o", default=None, help="输出路径")
    p.add_argument("--size", "-n", type=int, default=None, help="数量（默认读 EVAL_TESTSET_SIZE）")
    args = p.parse_args()

    cfg = get_eval_config()
    if args.size:
        cfg.eval_testset_size = args.size

    gen = SingleHopGenerator(cfg)
    result = gen.generate(output_path=args.output)
    print(f"生成单跳测试集: {len(result)} 条")
    for item in result[:3]:
        print(f"  [{item['query_id']}] {item['query'][:80]}...")
    if len(result) > 3:
        print(f"  ... 共 {len(result)} 条")
