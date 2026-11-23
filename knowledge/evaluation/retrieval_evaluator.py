"""RAGAS 检索评估器

加载测试集，逐条执行检索管道，收集 RAGAS 所需数据，
使用 LLM-as-judge 计算四项指标：
  - Faithfulness：答案是否基于检索上下文（反幻觉）
  - Answer Relevancy：答案是否切题
  - Context Precision：检索的 chunks 中有多少真正相关
  - Context Recall：关键信息是否都被检索到
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

# ragas 兼容性修复（须在任何 ragas 导入之前）
from knowledge.evaluation import _patch_ragas_vertexai_import  # noqa: F401

from knowledge.evaluation.config import get_eval_config

logger = logging.getLogger("eval.evaluator")


class RetrievalEvaluator:
    """基于 RAGAS 的检索质量评估器。

    对测试集中每条 query 执行完整检索管道，收集：
      - user_input: 用户问题
      - retrieved_contexts: 检索到的文档文本列表
      - response: LLM 生成的答案
      - reference: 参考答案（ground truth）

    然后使用 RAGAS LLM-as-judge 计算多维指标。
    """

    def __init__(self, config=None):
        self.config = config or get_eval_config()

    def evaluate(self, testset_path: str) -> Dict[str, Any]:
        """运行完整评估。

        Returns:
            评估报告 dict，含 summary / per_query / ragas_score。
        """
        # 1. 加载测试集
        items = self._load_testset(testset_path)
        logger.info(f"加载测试集: {len(items)} 条查询")

        # 2. 批量执行检索管道，收集 RAGAS 输入数据
        logger.info("执行检索管道...")
        records = self._run_pipeline(items)

        # 3. RAGAS 评估
        logger.info("RAGAS LLM-as-judge 评估中...")
        ragas_result = self._run_ragas_evaluate(records)

        # 4. 构建报告
        per_query = self._build_per_query(items, records, ragas_result)
        summary = self._compute_summary(ragas_result)

        report = {
            "timestamp": datetime.now().isoformat(),
            "testset": testset_path,
            "num_queries": len(items),
            "summary": summary,
            "per_query": per_query,
            "config_snapshot": self._snapshot_config(),
        }

        return report

    def evaluate_and_save(self, testset_path: str, output_path: str = None) -> str:
        """评估并保存报告。"""
        report = self.evaluate(testset_path)

        if output_path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            reports_dir = Path(self.config.eval_reports_dir)
            reports_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(reports_dir / f"eval_{ts}.json")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        self._print_summary(report["summary"])
        logger.info(f"评估报告已保存: {output_path}")
        return output_path

    # ------------------------------------------------------------------ #
    #  数据加载                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _load_testset(path: str) -> List[Dict]:
        with open(path, "r", encoding="utf-8") as f:
            items = json.load(f)
        for item in items:
            if "query" not in item:
                raise ValueError(f"测试集记录缺少 query 字段: {item}")
        return items

    # ------------------------------------------------------------------ #
    #  批量执行检索管道                                                       #
    # ------------------------------------------------------------------ #

    def _run_pipeline(self, items: List[Dict]) -> List[Dict]:
        """逐条执行查询管道，收集 RAGAS 输入数据。"""
        try:
            from knowledge.processor.query_process.main_graph import query_app
        except ImportError as e:
            logger.error(f"无法导入查询图: {e}")
            return []

        records = []
        for i, item in enumerate(items):
            query = item["query"]
            begin = time.time()

            try:
                state = {
                    "original_query": query,
                    "session_id": f"eval_{item.get('query_id', i)}",
                    "task_id": f"eval_task_{i}",
                    "is_stream": False,
                }
                result = query_app.invoke(state)

                # 提取检索上下文（reranked_docs 的 text 字段）
                reranked = result.get("reranked_docs", []) or []
                retrieved_contexts = [
                    d.get("text", "") for d in reranked if d.get("text", "").strip()
                ]
                response = result.get("answer", "") or ""

            except Exception as e:
                logger.warning(f"查询失败 [{item.get('query_id', i)}]: {e}")
                retrieved_contexts = []
                response = ""

            elapsed = time.time() - begin

            records.append({
                "query_id": item.get("query_id", f"q{i}"),
                "user_input": query,
                "retrieved_contexts": retrieved_contexts,
                "response": response,
                "reference": item.get("reference", ""),
                "elapsed_s": round(elapsed, 2),
            })

            if (i + 1) % 5 == 0 or i == len(items) - 1:
                logger.info(f"检索进度: {i + 1}/{len(items)}")

        return records

    # ------------------------------------------------------------------ #
    #  RAGAS 评估                                                          #
    # ------------------------------------------------------------------ #

    def _run_ragas_evaluate(self, records: List[Dict]) -> Dict:
        """使用 RAGAS LLM-as-judge 计算指标。"""
        if not records:
            return {}

        from ragas import evaluate, EvaluationDataset
        # 使用旧版指标 API（ragas 0.4.x collections 指标与 evaluate() 不兼容）
        from ragas.metrics import (
            faithfulness,
            context_precision,
            context_recall,
        )

        # 构建 RAGAS 评估 LLM（ragas 0.4.x 使用 llm_factory）
        from openai import OpenAI
        from ragas.llms import llm_factory
        from knowledge.processor.import_process.config import get_config

        cfg = get_config()
        # 评估专用 API（优先 EVAL_API_KEY，留空则复用管线 DashScope 凭证）
        import os as _os
        eval_api_key = _os.getenv("EVAL_API_KEY") or cfg.openai_api_key
        eval_api_base = _os.getenv("EVAL_API_BASE") or cfg.openai_api_base
        judge_client = OpenAI(api_key=eval_api_key, base_url=eval_api_base)
        judge_model = self.config.eval_judge_model or cfg.item_model or "deepseek-chat"
        evaluator_llm = llm_factory(
            judge_model, client=judge_client,
            max_tokens=4096,
        )

        # 构建 EvaluationDataset
        dataset_records = []
        for rec in records:
            dataset_records.append({
                "user_input": rec["user_input"],
                "retrieved_contexts": rec["retrieved_contexts"] or [""],
                "response": rec["response"] or "",
                "reference": rec["reference"] or "",
            })

        dataset = EvaluationDataset.from_list(dataset_records)

        try:
            result = evaluate(
                dataset=dataset,
                metrics=[faithfulness, context_precision, context_recall],
                llm=evaluator_llm,
                batch_size=4,  # 限制并发避免 DashScope 限流
            )
            # EvaluationResult is dict-like, e.g. {"faithfulness": 0.85, ...}
            # 部分评分任务可能因限流失败，跳过失败的取均值
            scores = {}
            try:
                result_df = result.to_pandas()
                for col in result_df.columns:
                    if col in ("user_input", "retrieved_contexts", "response", "reference"):
                        continue
                    vals = result_df[col].dropna().tolist()
                    vals = [v for v in vals if v is not None]
                    if vals:
                        scores[col] = round(sum(vals) / len(vals), 4)
                if not scores:
                    scores = {"note": "all jobs failed (likely rate limited)"}
            except Exception as parse_err:
                # Fallback: try raw dict
                try:
                    raw = dict(result)
                    for key, val in raw.items():
                        if hasattr(val, "tolist"):
                            vals = [v for v in val.tolist() if v is not None]
                            if vals:
                                scores[key] = round(sum(vals) / len(vals), 4)
                        elif isinstance(val, (int, float)):
                            scores[key] = round(float(val), 4)
                except Exception:
                    scores = {"error": str(parse_err)}
            logger.info(f"RAGAS 评估完成: {scores}")
            return scores
        except Exception as e:
            logger.error(f"RAGAS 评估失败: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}

    # ------------------------------------------------------------------ #
    #  报告构建                                                             #
    # ------------------------------------------------------------------ #

    def _build_per_query(
        self, items: List[Dict], records: List[Dict], ragas_result: Dict,
    ) -> List[Dict]:
        """构建逐条查询结果。"""
        per_query = []
        # RAGAS 逐条分数在 result DataFrame 中
        df = None
        if "error" not in ragas_result:
            try:
                df = ragas_result.to_pandas()
            except Exception:
                pass

        for i, (item, rec) in enumerate(zip(items, records)):
            entry = {
                "query_id": rec["query_id"],
                "query": item["query"],
                "reference": item.get("reference", ""),
                "response": rec["response"][:500],
                "num_retrieved": len(rec["retrieved_contexts"]),
                "elapsed_s": rec["elapsed_s"],
            }
            # 附加 RAGAS 逐条分数
            if df is not None and i < len(df):
                for col in df.columns:
                    val = df.iloc[i][col]
                    if isinstance(val, float):
                        entry[col] = round(val, 4)
            per_query.append(entry)

        return per_query

    @staticmethod
    def _compute_summary(ragas_result: Dict) -> Dict[str, float]:
        """从 RAGAS 结果中提取汇总指标。"""
        summary = {}
        for key, val in ragas_result.items():
            if key == "error":
                summary["error"] = val
            elif hasattr(val, "tolist"):
                values = [v for v in val.tolist() if v is not None]
                if values:
                    summary[key] = round(sum(values) / len(values), 4)
            elif isinstance(val, (int, float)):
                summary[key] = round(float(val), 4)
            elif key == "note":
                summary["note"] = val
        return summary

    @staticmethod
    def _snapshot_config() -> Dict[str, Any]:
        try:
            from knowledge.processor.query_process.config import get_config
            qc = get_config()
            return {
                "rrf_k": qc.rrf_k,
                "rrf_kg_weight": qc.rrf_kg_weight,
                "rrf_max_results": qc.rrf_max_results,
                "embedding_search_limit": qc.embedding_search_limit,
                "hyde_search_limit": qc.hyde_search_limit,
                "rerank_max_top_k": qc.rerank_max_top_k,
            }
        except Exception:
            return {}

    @staticmethod
    def _print_summary(summary: Dict):
        print("\n" + "=" * 60)
        print("RAGAS 评估结果")
        print("=" * 60)
        for metric, score in summary.items():
            if metric == "error":
                print(f"  ERROR: {score}")
            else:
                print(f"  {metric}: {score:.4f}")
        print("-" * 60)
        print("  指标说明:")
        print("    faithfulness       — 答案是否基于检索内容（反幻觉）")
        print("    context_precision  — 检索的文档是否相关")
        print("    context_recall     — 关键信息是否都召回")
        print("  注: AnswerRelevancy 需要额外 embedding 模型，暂未启用")
        print("=" * 60)


# ================================================================== #
#  测试入口                                                              #
# ================================================================== #

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="RAGAS 检索评估器")
    parser.add_argument("--testset", "-t", required=True, help="测试集 JSON 文件路径")
    parser.add_argument("--output", "-o", default=None, help="评估报告输出路径")
    args = parser.parse_args()

    evaluator = RetrievalEvaluator()
    evaluator.evaluate_and_save(args.testset, args.output)
