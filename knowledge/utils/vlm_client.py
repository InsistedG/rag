"""VLM 视觉语言模型客户端"""

import base64
import logging
import os
import time
from pathlib import Path

import httpx
from openai import APIConnectionError, APITimeoutError, APIStatusError, OpenAI, RateLimitError

from knowledge.processor.import_process.config import get_config

logger = logging.getLogger("import.vlm")

# 瞬时网络/代理抖动时重试（DashScope 经 Clash fake-ip 时常见 Connection error）
_MAX_RETRIES = 4
_RETRY_BASE_DELAY = 2.0
_CONNECT_TIMEOUT = 20.0
_REQUEST_TIMEOUT = 120.0


def _is_retryable(exc: BaseException) -> bool:
    """判断是否为可重试的瞬时错误（连接/超时/限流/5xx）。"""
    if isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError)):
        return True
    if isinstance(exc, APIStatusError) and exc.status_code in {408, 409, 425, 429, 500, 502, 503, 504}:
        return True
    if isinstance(exc, (httpx.TransportError, httpx.TimeoutException, TimeoutError, ConnectionError, OSError)):
        return True
    msg = str(exc).lower()
    return any(
        key in msg
        for key in (
            "connection error",
            "connect error",
            "timed out",
            "timeout",
            "temporarily unavailable",
            "connection reset",
            "remoteprotocolerror",
            "server disconnected",
        )
    )


class VLMClient:
    """视觉语言模型客户端，用于描述图片内容"""

    SYSTEM_PROMPT = (
        "你是电子产品说明书解析专家。你的任务是为产品手册中的图片生成检索友好的alt文本。"
        "描述必须详细、准确，包含图片中的所有编号标注、接口名称、按钮位置、指示灯状态等信息，"
        "使文本描述可以替代图片被RAG系统检索。"
    )

    def __init__(self):
        config = get_config()
        # VLM 专用配置优先（支持独立于管线的 API），留空回退管线配置
        vl_key = (os.getenv("VL_API_KEY") or config.openai_api_key or "not-needed").strip()
        vl_base = (os.getenv("VL_API_BASE") or config.openai_api_base or "").strip()
        self.model = (config.vl_model or "qwen3-vl-flash").strip()
        self.rpm = config.requests_per_minute
        self._last_call_time = 0.0

        if not vl_base:
            logger.warning("VL_API_BASE / OPENAI_API_BASE 未配置，VLM 调用很可能失败")
        logger.info(f"VLM 初始化: base={vl_base}, model={self.model}")

        # 显式 httpx：拉长 connect 超时；不叠加 SDK 重试（由本类控制）
        # trust_env=True：尊重系统/Clash 代理；TUN 模式下也兼容
        timeout = httpx.Timeout(
            connect=_CONNECT_TIMEOUT,
            read=_REQUEST_TIMEOUT,
            write=_REQUEST_TIMEOUT,
            pool=_CONNECT_TIMEOUT,
        )
        http_client = httpx.Client(timeout=timeout, trust_env=True)

        self.client = OpenAI(
            base_url=vl_base or None,
            api_key=vl_key,
            timeout=_REQUEST_TIMEOUT,
            max_retries=0,
            http_client=http_client,
        )

    def describe_image(self, image_path: str) -> str:
        """
        使用 VLM 详细描述图片内容，为 RAG 检索提供结构化文本。

        Args:
            image_path: 图片文件路径

        Returns:
            图片的结构化中文描述
        """
        image_path = Path(image_path)
        if not image_path.exists():
            logger.warning(f"图片不存在: {image_path}")
            return ""

        suffix = image_path.suffix.lower()
        media_type_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
        }
        media_type = media_type_map.get(suffix, "image/jpeg")

        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        data_url = f"data:{media_type};base64,{image_data}"

        user_prompt = (
            "请详细描述图片内容。若是产品外观或接口示意图：\n"
            "1. 列出每个编号对应的接口/按钮名称（如 1=电源键、2=USB-C接口）\n"
            "2. 描述各组件的相对位置（左侧/右侧/背面/正面）\n"
            "3. 抄录图中所有文字标签和警告标识\n"
            "若是普通插图或流程图，用2-3句话概括其含义。"
        )

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    },
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]

        last_error: Exception | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            self._rate_limit()
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=500,
                )
                content = response.choices[0].message.content or ""
                description = content.strip()
                logger.info(f"VLM 描述: {image_path.name} -> {description[:80]}...")
                return description
            except Exception as e:
                last_error = e
                if _is_retryable(e) and attempt < _MAX_RETRIES:
                    delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
                    logger.warning(
                        f"VLM 瞬时失败 ({attempt}/{_MAX_RETRIES}): {image_path.name}, "
                        f"{type(e).__name__}: {e}; {delay:.1f}s 后重试"
                    )
                    time.sleep(delay)
                    continue
                if _is_retryable(e):
                    break
                logger.error(
                    f"VLM 描述失败: {image_path.name}, "
                    f"{type(e).__name__}: {e}"
                )
                return ""

        logger.error(
            f"VLM 描述失败（已重试 {_MAX_RETRIES} 次）: {image_path.name}, "
            f"{type(last_error).__name__}: {last_error}"
        )
        return ""

    def _rate_limit(self):
        """简单的速率限制"""
        now = time.time()
        min_interval = 60.0 / max(self.rpm, 1)
        elapsed = now - self._last_call_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_call_time = time.time()
