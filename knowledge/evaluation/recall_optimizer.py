"""召回超参数优化器（RAGAS 版）

对检索管道中的关键参数进行网格/随机搜索，
使用 RAGAS 指标（默认 faithfulness）作为优化目标。
"""

import copy
import itertools
import json
import logging
import random
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from knowledge.evaluation.config import get_eval_config

logger = logging.getLogger("eval.optimizer")

# 可调超参数搜索空间（全网格 144 组，用随机搜索采样以减少时间）
SEARCH_SPACE = {
    "rrf_k": [30, 60, 100],
    "rrf_kg_weight": [0.5, 0.7, 1.0],
    "rerank_max_top_k": [5, 8, 10, 15],
    "embedding_search_limit": [10, 15, 20],
    "hyde_search_limit": [10, 15, 20],
}


class RecallOptimizer:
    """检索超参数优化器。

    对 SEARCH_SPACE 中的参数进行搜索，
    使用 RAGAS 指标评估每组组合，找到最优配置。
    """

    def __init__(self, config=None):
        self.config = config or get_eval_config()

    def optimize(self, testset_path: str) -> Dict[str, Any]:
        """运行超参数优化。

        Args:
            testset_path: 测试集 JSON 文件路径。

        Returns:
            优化结果 dict。
        """
        logger.info(
            f"超参数优化开始: method={self.config.eval_optimization_method}, "
            f"target={self.config.eval_target_metric}"
        )

        # 1. 生成候选参数组合
        trials = self._generate_trials()
        logger.info(f"候选参数组合: {len(trials)} 组")

        # 2. 逐组评估
        results = []
        best_score = -1.0
        best_params = None

        for i, params in enumerate(trials):
            logger.info(f"[{i + 1}/{len(trials)}] {self._params_str(params)}")
            begin = time.time()

            score = self._evaluate_params(params, testset_path)
            elapsed = time.time() - begin

            result = {
                "trial": i + 1,
                "params": params,
                "score": round(score, 4),
                "elapsed_s": round(elapsed, 1),
            }
            results.append(result)
            logger.info(f"  -> {self.config.eval_target_metric}={score:.4f} ({elapsed:.1f}s)")

            if score > best_score:
                best_score = score
                best_params = copy.deepcopy(params)

        # 3. 保存结果
        output = {
            "timestamp": datetime.now().isoformat(),
            "testset": testset_path,
            "method": self.config.eval_optimization_method,
            "target_metric": self.config.eval_target_metric,
            "num_trials": len(trials),
            "best_params": best_params,
            "best_score": round(best_score, 4),
            "all_trials": sorted(results, key=lambda r: r["score"], reverse=True),
        }

        report_path = self._save_report(output)
        self._print_result(output)
        logger.info(f"优化报告已保存: {report_path}")
        return output

    # ------------------------------------------------------------------ #
    #  参数组合生成                                                          #
    # ------------------------------------------------------------------ #

    def _generate_trials(self) -> List[Dict[str, Any]]:
        method = self.config.eval_optimization_method
        if method == "random":
            return self._random_search()
        # 默认网格搜索
        return self._grid_search()

    def _grid_search(self) -> List[Dict]:
        keys = list(SEARCH_SPACE.keys())
        values = list(SEARCH_SPACE.values())
        trials = []
        for combo in itertools.product(*values):
            trials.append(dict(zip(keys, combo)))
        logger.info(f"网格搜索: {len(trials)} 组 ({len(keys)} 参数)")
        return trials

    def _random_search(self) -> List[Dict]:
        n = min(self.config.eval_optimization_trials, len(self._grid_search()))
        keys = list(SEARCH_SPACE.keys())
        values = list(SEARCH_SPACE.values())
        trials = []
        seen = set()
        while len(trials) < n:
            combo = tuple(random.choice(v) for v in values)
            if combo not in seen:
                seen.add(combo)
                trials.append(dict(zip(keys, combo)))
        logger.info(f"随机搜索: {len(trials)} 组")
        return trials

    # ------------------------------------------------------------------ #
    #  单组参数评估                                                          #
    # ------------------------------------------------------------------ #

    def _evaluate_params(self, params: Dict, testset_path: str) -> float:
        """用一组参数运行 RAGAS 评估，返回目标指标分数。"""
        try:
            from knowledge.processor.query_process.config import get_config
        except ImportError as e:
            logger.error(f"无法导入 QueryConfig: {e}")
            return 0.0

        qc = get_config()
        backup = {}
        for key, val in params.items():
            backup[key] = getattr(qc, key, None)
            setattr(qc, key, val)

        try:
            from knowledge.evaluation.retrieval_evaluator import RetrievalEvaluator
            evaluator = RetrievalEvaluator(self.config)
            report = evaluator.evaluate(testset_path)

            target = self.config.eval_target_metric
            score = report.get("summary", {}).get(target, 0.0)
            return score if isinstance(score, (int, float)) else 0.0
        finally:
            for key, val in backup.items():
                setattr(qc, key, val)

    # ------------------------------------------------------------------ #
    #  输出                                                                #
    # ------------------------------------------------------------------ #

    def _save_report(self, output: Dict) -> str:
        reports_dir = Path(self.config.eval_reports_dir)
        reports_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = str(reports_dir / f"optimize_{ts}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        return path

    @staticmethod
    def _params_str(params: Dict) -> str:
        return ", ".join(f"{k}={v}" for k, v in sorted(params.items()))

    @staticmethod
    def _print_result(output: Dict):
        print("\n" + "=" * 60)
        print("超参数优化结果 (RAGAS)")
        print("=" * 60)
        print(f"  目标指标:  {output['target_metric']}")
        print(f"  最优分数:  {output['best_score']:.4f}")
        print("  最优参数:")
        for k, v in sorted(output.get("best_params", {}).items()):
            print(f"    {k} = {v}")
        print("-" * 60)
        print("  Top 5:")
        for r in output.get("all_trials", [])[:5]:
            print(
                f"    [{r['trial']:2d}] score={r['score']:.4f}  "
                f"{RecallOptimizer._params_str(r['params'])}"
            )
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

    parser = argparse.ArgumentParser(description="RAG 召回超参数优化器")
    parser.add_argument("--testset", "-t", required=True, help="测试集 JSON 文件路径")
    parser.add_argument("--method", "-m", default=None, choices=["grid", "random"])
    parser.add_argument("--target", default=None, help="目标指标（默认 faithfulness）")
    args = parser.parse_args()

    cfg = get_eval_config()
    if args.method:
        cfg.eval_optimization_method = args.method
    if args.target:
        cfg.eval_target_metric = args.target

    optimizer = RecallOptimizer(cfg)
    optimizer.optimize(args.testset)
