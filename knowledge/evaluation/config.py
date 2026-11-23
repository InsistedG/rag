"""评估流程配置管理"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import os

from dotenv import load_dotenv
from pathlib import Path
load_dotenv(Path(__file__).parent.parent / ".env", override=True)


def _parse_k_values() -> List[int]:
    raw = os.getenv("EVAL_K_VALUES", "1,3,5,10")
    return [int(k.strip()) for k in raw.split(",") if k.strip().isdigit()]


@dataclass
class EvalConfig:
    """评估流程配置。"""

    # ==================== 测试集生成 ====================
    eval_testset_size: int = field(
        default_factory=lambda: int(os.getenv("EVAL_TESTSET_SIZE", "30"))
    )
    eval_sample_collection: str = field(
        default_factory=lambda: os.getenv(
            "EVAL_SAMPLE_COLLECTION",
            os.getenv("CHUNKS_COLLECTION", "kb_chunks_v2"),
        )
    )
    eval_sample_limit: int = field(
        default_factory=lambda: int(os.getenv("EVAL_SAMPLE_LIMIT", "100"))
    )
    eval_llm_model: str = field(
        default_factory=lambda: os.getenv(
            "EVAL_LLM_MODEL",
            os.getenv("ITEM_MODEL", "qwen-flash"),
        )
    )
    # RAGAS 评估用 LLM（需要较强的推理能力来做 judge）
    eval_judge_model: str = field(
        default_factory=lambda: os.getenv(
            "EVAL_JUDGE_MODEL",
            os.getenv("EVAL_LLM_MODEL",
                os.getenv("LLM_DEFAULT_MODEL", "qwen3.7-max")),
        )
    )

    # ==================== 评估 ====================
    eval_k_values: List[int] = field(default_factory=_parse_k_values)

    # ==================== 优化 ====================
    eval_optimization_trials: int = field(
        default_factory=lambda: int(os.getenv("EVAL_OPTIMIZATION_TRIALS", "36"))
    )
    eval_optimization_method: str = field(
        default_factory=lambda: os.getenv("EVAL_OPTIMIZATION_METHOD", "grid")
    )
    # RAGAS 目标指标: faithfulness / answer_relevancy / context_precision / context_recall
    eval_target_metric: str = field(
        default_factory=lambda: os.getenv("EVAL_TARGET_METRIC", "faithfulness")
    )

    # ==================== 输出路径 ====================
    eval_data_dir: str = field(
        default_factory=lambda: os.getenv(
            "EVAL_DATA_DIR",
            str(Path(__file__).resolve().parent / "data"),
        )
    )
    eval_reports_dir: str = field(
        default_factory=lambda: os.getenv(
            "EVAL_REPORTS_DIR",
            str(Path(__file__).resolve().parent / "reports"),
        )
    )

    @classmethod
    def from_env(cls) -> "EvalConfig":
        return cls()


_config: Optional[EvalConfig] = None


def get_eval_config() -> EvalConfig:
    global _config
    if _config is None:
        _config = EvalConfig.from_env()
    return _config
