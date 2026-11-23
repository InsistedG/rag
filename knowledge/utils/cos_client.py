"""腾讯云 COS 客户端工具

基于 cos-python-sdk-v5，替代 MinIO 进行图片存储。
暴露与 MinioClient 相同的接口，便于无缝替换。
"""

import logging
from pathlib import Path

from qcloud_cos import CosConfig, CosS3Client

from knowledge.processor.import_process.config import get_config

logger = logging.getLogger("import.cos")


class CosClient:
    """腾讯云 COS 客户端封装，暴露与 MinioClient 相同的接口"""

    def __init__(self):
        config = get_config()
        self.secret_id = config.cos_secret_id
        self.secret_key = config.cos_secret_key
        self.region = config.cos_region or "ap-guangzhou"
        self.bucket = config.cos_bucket or "shopkeeper-images"

        if not self.secret_id or not self.secret_key:
            logger.warning("COS 凭证未配置，上传功能将不可用")

        cos_config = CosConfig(
            Region=self.region,
            SecretId=self.secret_id,
            SecretKey=self.secret_key,
        )
        self._client = CosS3Client(cos_config)
        self._ensure_bucket()

    def _ensure_bucket(self):
        """确保 bucket 存在并设为公有读。

        注意：COS 通过控制台或 create_bucket 设置公有读。
        这里仅做存在性检查，bucket 需提前在控制台创建并设为公有读。
        """
        try:
            self._client.head_bucket(Bucket=self.bucket)
            logger.info(f"COS bucket 已存在: {self.bucket}")
        except Exception:
            try:
                self._client.create_bucket(
                    Bucket=self.bucket,
                    ACL="public-read",
                )
                logger.info(f"创建 COS bucket: {self.bucket}（公有读）")
            except Exception as e:
                logger.warning(f"检查/创建 bucket 失败（可能已存在或权限不足）: {e}")

    def upload_file(self, local_path: str, object_name: str = None) -> str:
        """上传文件到 COS，返回公开访问 URL。

        Args:
            local_path: 本地文件路径
            object_name: COS 中的对象键名，默认使用文件名

        Returns:
            文件的公开访问 URL（CDN 加速域名或默认域名）
        """
        if object_name is None:
            object_name = Path(local_path).name

        # 根据扩展名推断 Content-Type
        ext = Path(local_path).suffix.lower()
        content_type_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".bmp": "image/bmp",
            ".webp": "image/webp",
            ".svg": "image/svg+xml",
        }
        content_type = content_type_map.get(ext, "application/octet-stream")

        self._client.put_object(
            Bucket=self.bucket,
            Body=open(local_path, "rb"),
            Key=object_name,
            ACL="public-read",
            ContentType=content_type,
            EnableMD5=False,
        )

        # 公开访问 URL
        url = f"https://{self.bucket}.cos.{self.region}.myqcloud.com/{object_name}"
        logger.info(f"上传成功: {local_path} -> {url}")
        return url
