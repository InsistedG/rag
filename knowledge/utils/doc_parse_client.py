"""MinerU 官方 API 文档解析客户端

通过 MinerU 云端 API 将 PDF 文件解析为 Markdown 格式，替代本地 MinerU CLI。

API 流程（3 步）：
  1. POST /api/v4/file-urls/batch — 获取预签名上传 URL + batch_id
  2. HTTP PUT 上传 PDF 文件（系统自动提交解析任务）
  3. 轮询 GET /api/v4/extract-results/batch/{batch_id} — 等待完成
  4. 下载 ZIP 并提取 full.md + images/

参考文档: https://mineru.net/doc/docs/
"""

import io
import logging
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path

import requests

from knowledge.processor.import_process.config import get_config

logger = logging.getLogger("import.doc_parse")

MINERU_API_BASE = "https://mineru.net/api/v4"


class DocParseError(Exception):
    """文档解析异常"""

    def __init__(self, message: str, step: str = ""):
        self.step = step
        super().__init__(f"[{step}] {message}" if step else message)


class DocParseClient:
    """MinerU 云端文档解析客户端。

    使用 MinerU 批量上传 API 实现本地 PDF → Markdown + 图片解析。
    上传后系统自动提交任务，无需手动调用 extract/task。
    """

    POLL_INTERVAL = 3
    POLL_TIMEOUT = 600

    def __init__(self):
        config = get_config()
        self.api_token = config.doc_parse_api_token or ""
        if not self.api_token:
            logger.warning("MINERU_API_TOKEN 未配置，文档解析 API 将不可用")

    # ------------------------------------------------------------------ #
    #  公开接口                                                             #
    # ------------------------------------------------------------------ #

    def parse_pdf(self, pdf_path: str, output_dir: str = "") -> str:
        """将本地 PDF 文件解析为 Markdown 文本。

        Args:
            pdf_path: PDF 文件本地路径。
            output_dir: 图片输出目录。不为空时 ZIP 中的 images/ 解压到此目录下。

        Returns:
            解析后的 Markdown 文本。

        Raises:
            DocParseError: 解析过程中任意步骤失败。
            FileNotFoundError: PDF 文件不存在。
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")

        begin = time.time()

        # 1. 获取上传 URL 并上传 PDF
        logger.info("步骤1/3: 上传 PDF 到 MinerU")
        batch_id = self._upload_via_batch(pdf_path)

        # 2. 轮询等待解析完成（上传后系统自动提交）
        logger.info(f"步骤2/3: 等待解析完成 (batch_id={batch_id})")
        full_zip_url = self._poll_batch_result(batch_id, pdf_path.name)

        # 3. 下载 ZIP，提取 Markdown 和图片
        logger.info("步骤3/3: 下载并解压解析结果")
        md_content = self._extract_zip(full_zip_url, output_dir)

        elapsed = time.time() - begin
        img_info = ""
        if output_dir:
            img_dir = Path(output_dir) / "images"
            if img_dir.exists():
                img_count = len(list(img_dir.iterdir()))
                img_info = f", {img_count} 张图片"
        logger.info(
            f"文档解析完成: {pdf_path.name} 耗时 {elapsed:.1f}s, "
            f"Markdown {len(md_content)} 字符{img_info}"
        )
        return md_content

    # ------------------------------------------------------------------ #
    #  步骤1: 批量上传                                                        #
    # ------------------------------------------------------------------ #

    def _upload_via_batch(self, pdf_path: Path) -> str:
        """获取上传 URL 并 PUT 上传文件，返回 batch_id。"""
        headers = self._auth_header()
        headers["Content-Type"] = "application/json"

        # 1a. 申请上传 URL
        body = {"files": [{"name": pdf_path.name}]}
        resp = self._request(
            "POST", f"{MINERU_API_BASE}/file-urls/batch",
            headers=headers, json_data=body, step="申请上传URL",
        )
        data = resp.json()
        self._check_code(data, "申请上传URL")

        batch_id = data["data"]["batch_id"]
        file_urls = data.get("data", {}).get("file_urls", [])
        if not file_urls:
            raise DocParseError("未获取到上传 URL", "申请上传URL")

        # 1b. PUT 上传（不设 Content-Type）
        with open(pdf_path, "rb") as f:
            file_data = f.read()
        put_resp = requests.put(file_urls[0], data=file_data, timeout=120)
        if put_resp.status_code not in (200, 201, 204):
            raise DocParseError(
                f"文件上传失败: HTTP {put_resp.status_code}", "上传文件"
            )
        logger.info(f"文件上传成功: {pdf_path.name}")

        return batch_id

    # ------------------------------------------------------------------ #
    #  步骤2: 轮询批量结果                                                     #
    # ------------------------------------------------------------------ #

    def _poll_batch_result(self, batch_id: str, file_name: str) -> str:
        """轮询批量解析结果，返回 full_zip_url。"""
        headers = self._auth_header()
        url = f"{MINERU_API_BASE}/extract-results/batch/{batch_id}"

        start = time.time()
        while True:
            time.sleep(self.POLL_INTERVAL)
            resp = self._request("GET", url, headers=headers, step="查询解析结果")
            data = resp.json()
            self._check_code(data, "查询解析结果")

            results = data.get("data", {}).get("extract_result", [])
            if not results:
                raise DocParseError("响应中缺少 extract_result", "查询解析结果")

            file_result = results[0]
            state = file_result.get("state", "")

            if state == "done":
                full_zip_url = file_result.get("full_zip_url", "")
                if not full_zip_url:
                    raise DocParseError("full_zip_url 为空", "查询解析结果")
                logger.info(f"解析完成: {file_name}")
                return full_zip_url

            if state == "failed":
                err_msg = file_result.get("err_msg", "未知错误")
                raise DocParseError(f"解析失败: {err_msg}", "轮询")

            progress = file_result.get("extract_progress", {})
            if progress:
                cur = progress.get("extracted_pages", 0)
                total = progress.get("total_pages", 0)
                if total:
                    logger.info(f"解析进度: {cur}/{total} 页")

            if time.time() - start > self.POLL_TIMEOUT:
                raise DocParseError(f"解析超时 ({self.POLL_TIMEOUT}s)", "轮询")

    # ------------------------------------------------------------------ #
    #  步骤3: 下载 ZIP 并提取 Markdown + 图片                                  #
    # ------------------------------------------------------------------ #

    def _extract_zip(self, full_zip_url: str, output_dir: str) -> str:
        """下载结果 ZIP，提取 full.md 文本和 images/ 图片。

        Args:
            full_zip_url: MinerU 结果 ZIP 下载地址。
            output_dir: 图片输出根目录。不为空时解压 images/ 到 {output_dir}/images/。

        Returns:
            full.md 的文本内容。
        """
        zip_bytes = self._download_zip(full_zip_url)

        try:
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                # 提取 Markdown
                md_content = ""
                if "full.md" in zf.namelist():
                    md_content = zf.read("full.md").decode("utf-8")
                else:
                    md_files = [n for n in zf.namelist() if n.endswith(".md")]
                    if md_files:
                        md_files.sort(
                            key=lambda n: zf.getinfo(n).file_size, reverse=True
                        )
                        md_content = zf.read(md_files[0]).decode("utf-8")
                    else:
                        parts = []
                        for name in zf.namelist():
                            try:
                                parts.append(zf.read(name).decode("utf-8"))
                            except Exception:
                                pass
                        md_content = "\n\n".join(parts)

                # 提取图片
                if output_dir:
                    image_names = [
                        n for n in zf.namelist()
                        if n.startswith("images/") and not n.endswith("/")
                    ]
                    if image_names:
                        images_dir = Path(output_dir) / "images"
                        images_dir.mkdir(parents=True, exist_ok=True)
                        for name in image_names:
                            target = images_dir / Path(name).name
                            if not target.exists():
                                target.write_bytes(zf.read(name))
                        logger.info(f"提取 {len(image_names)} 张图片到 {images_dir}")

                return md_content
        except zipfile.BadZipFile:
            return zip_bytes.decode("utf-8", errors="replace")

    def _download_zip(self, full_zip_url: str) -> bytes:
        """下载 ZIP 文件。Python SSL 失败时回退到 curl。"""
        # 方式 1: Python requests
        for attempt in range(3):
            try:
                resp = self._request(
                    "GET", full_zip_url,
                    step="下载解析结果", verify_ssl=False,
                )
                if resp.status_code == 200:
                    return resp.content
            except DocParseError:
                if attempt < 2:
                    time.sleep(2)

        # 方式 2: curl 回退
        logger.info("Python SSL 下载失败，回退到 curl")
        return self._download_via_curl(full_zip_url)

    def _download_via_curl(self, url: str) -> bytes:
        """通过 curl 子进程下载文件。"""
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
                tmp_path = f.name

            subprocess.run(
                ["curl", "-k", "-L", "-s", "-o", tmp_path, url],
                timeout=120, check=True,
                capture_output=True, text=True,
            )

            with open(tmp_path, "rb") as f:
                content = f.read()
            if not content:
                raise DocParseError("curl 下载结果为空", "下载解析结果")
            return content
        except FileNotFoundError:
            raise DocParseError("系统未安装 curl", "下载解析结果")
        finally:
            if tmp_path:
                try:
                    import os
                    os.unlink(tmp_path)
                except Exception:
                    pass

    # ------------------------------------------------------------------ #
    #  工具方法                                                             #
    # ------------------------------------------------------------------ #

    def _auth_header(self) -> dict:
        return {"Authorization": f"Bearer {self.api_token}"}

    @staticmethod
    def _request(method: str, url: str, headers: dict = None,
                 json_data: dict = None, data=None, step: str = "",
                 timeout: int = 120, verify_ssl: bool = True) -> requests.Response:
        try:
            kwargs = dict(headers=headers, timeout=timeout)
            if not verify_ssl:
                import urllib3
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                kwargs["verify"] = False

            if method == "POST":
                kwargs["json"] = json_data
                resp = requests.post(url, **kwargs)
            elif method == "GET":
                resp = requests.get(url, **kwargs)
            elif method == "PUT":
                kwargs["data"] = data
                resp = requests.put(url, **kwargs)
            else:
                raise ValueError(f"不支持的 HTTP 方法: {method}")

            if resp.status_code >= 500:
                raise DocParseError(f"服务器错误: HTTP {resp.status_code}", step)
            return resp
        except requests.RequestException as e:
            raise DocParseError(str(e), step) from e

    @staticmethod
    def _check_code(data: dict, step: str):
        code = data.get("code")
        if code is not None and code != 0:
            msg = data.get("msg") or data.get("message") or "未知错误"
            raise DocParseError(f"API 返回错误 (code={code}): {msg}", step)
