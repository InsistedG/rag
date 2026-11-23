import json
import time
from pathlib import Path
from typing import Tuple
import subprocess

from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.config import get_config
from knowledge.processor.import_process.exceptions import ValidationError, FileProcessingError, PdfConversionError
from knowledge.processor.import_process.state import ImportGraphState


class PdfToMdNode(BaseNode):
    """
    PDF 转 Markdown 节点

    支持两种模式（由 DOC_PARSE_ENABLED 环境变量控制）：
      - API 模式: 调用 MinerU 官方云端 API 解析 PDF
      - 本地模式: 调用本地 MinerU CLI (mineru 命令)
    """

    name = "pdf_to_md_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        # 1. 校验输入路径
        import_file_path_obj, file_dir_obj = self._validate_state_input_path(state)

        # 2. 解析 PDF → Markdown（按配置选择 API 或本地 CLI）
        config = get_config()
        if config.doc_parse_enabled:
            md_content = self._execute_api_parse(import_file_path_obj, file_dir_obj)
            md_path = self._write_md_file(md_content, import_file_path_obj, file_dir_obj)
        else:
            process_code = self._execute_mineru(import_file_path_obj, file_dir_obj)
            if process_code != 0:
                raise PdfConversionError("执行MinerU解析PDF失败", self.name)
            md_path = self._find_mineru_output(file_dir_obj, import_file_path_obj, state)

        # 3. 更新 state
        state["md_path"] = str(md_path)
        return state

    # ------------------------------------------------------------------ #
    #  校验输入路径                                                          #
    # ------------------------------------------------------------------ #

    def _validate_state_input_path(self, state: ImportGraphState) -> Tuple[Path, Path]:
        self.log_step("step1", "[校验输入文件路径]")
        import_file_path = state.get("import_file_path", "")
        file_dir = state.get("file_dir", "")
        if not import_file_path:
            raise ValidationError("解析的文件不存在", self.name)
        import_file_path_obj = Path(import_file_path)
        if not import_file_path_obj.exists():
            raise FileProcessingError("解析的文件路径不存在", self.name)
        if not file_dir:
            file_dir = import_file_path_obj.parent

        file_dir_obj = Path(file_dir)
        self.logger.info(f"输入文件路径: {import_file_path_obj}, 文件目录: {file_dir_obj}")
        return import_file_path_obj, file_dir_obj

    # ------------------------------------------------------------------ #
    #  API 模式: MinerU 官方云端 API                                          #
    # ------------------------------------------------------------------ #

    def _execute_api_parse(self, import_file_path_obj: Path, file_dir_obj: Path) -> str:
        """调用 MinerU 官方 API 将 PDF 解析为 Markdown 文本，同时提取图片。"""
        self.log_step("step2", "[调用 MinerU API 解析 PDF]")

        from knowledge.utils.doc_parse_client import DocParseClient, DocParseError

        client = DocParseClient()
        # 图片输出到 {file_dir}/{file_title}/images/（与后续 _write_md_file 路径一致）
        file_title = import_file_path_obj.stem
        output_dir = str(file_dir_obj / file_title)

        try:
            md_content = client.parse_pdf(
                str(import_file_path_obj), output_dir=output_dir,
            )
            self.logger.info(
                f"API 解析成功: {import_file_path_obj.name}, "
                f"Markdown 长度: {len(md_content)} 字符"
            )
            return md_content
        except DocParseError as e:
            raise PdfConversionError(f"MinerU API 解析失败: {e}", self.name) from e

    # ------------------------------------------------------------------ #
    #  本地模式: MinerU CLI 子进程                                             #
    # ------------------------------------------------------------------ #

    def _execute_mineru(self, import_file_path_obj, file_dir_obj) -> int:
        self.log_step("step2", "[执行MinerU解析PDF]")
        cmd = [
            "mineru",
            "-p", str(import_file_path_obj),
            "-o", str(file_dir_obj),
            "--source", "local",
        ]
        begin = time.time()
        process = subprocess.Popen(
            args=cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            errors="replace",
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        for output in process.stdout:
            self.logger.info(f"执行MinerU产生的日志: {output.strip()}")
        process_code = process.wait()
        end = time.time()

        if process_code != 0:
            self.logger.info(
                f"执行MinerU解析PDF失败:{import_file_path_obj.name} 耗时{end-begin:.2f}s"
            )
        else:
            self.logger.info(
                f"执行MinerU解析PDF成功:{import_file_path_obj.name} 耗时{end-begin:.2f}s"
            )
        return process_code

    # ------------------------------------------------------------------ #
    #  输出路径处理                                                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _find_mineru_output(file_dir_obj: Path, import_file_path_obj: Path,
                            state: ImportGraphState) -> Path:
        """本地模式: 找到 MinerU CLI 输出的 MD 文件路径。"""
        file_title = state.get("file_title", "") or import_file_path_obj.stem
        md_output_path = file_dir_obj / file_title / "hybrid_auto" / f"{file_title}.md"
        if not md_output_path.exists():
            raise FileProcessingError(f"解析后的MD文件不存在: {md_output_path}", "pdf_to_md_node")
        return md_output_path

    @staticmethod
    def _write_md_file(md_content: str, import_file_path_obj: Path,
                       file_dir_obj: Path) -> Path:
        """API 模式: 将 Markdown 内容写入文件（模拟 MinerU 输出目录结构）。"""
        file_title = import_file_path_obj.stem
        output_dir = file_dir_obj / file_title
        output_dir.mkdir(parents=True, exist_ok=True)
        md_path = output_dir / f"{file_title}.md"
        md_path.write_text(md_content, encoding="utf-8")
        return md_path


if __name__ == "__main__":
    setup_logging()
    # 简单测试
    node = PdfToMdNode()
    test_state = {
        "import_file_path": r"E:\AI+Py\shopkeeper_brain\knowledge\processor\import_process\import_temp_dir\hak180产品安全手册.pdf",
        "file_dir": r"E:\AI+Py\shopkeeper_brain\knowledge\processor\import_process\output_temp_dir",
        "file_title": "万用表RS-12的使用",
    }
    try:
        result_state = node.process(test_state)
        print("处理后的状态:", json.dumps(s=result_state, indent=2, ensure_ascii=False))
    except ValidationError as e:
        print("验证错误:", e)