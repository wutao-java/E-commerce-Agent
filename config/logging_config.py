"""配置应用的控制台日志和轮转文件日志。"""

# 启用延迟解析类型注解。
from __future__ import annotations

# 导入标准日志、轮转文件处理器和路径处理能力。
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def configure_logging(level: str = "INFO", log_file: str | None = None) -> None:
    """初始化应用的根日志处理器。

    Args:
        level: 日志级别名称，不区分大小写。
        log_file: 可选的日志文件路径；为空时仅输出到控制台。

    Returns:
        None.
    """
    # 默认注册控制台处理器，确保未配置文件时仍有日志输出。
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    # 仅在明确提供日志路径时启用文件日志。
    if log_file:
        # 将字符串路径转换为跨平台路径对象。
        path = Path(log_file)
        # 自动创建日志目录，重复初始化时保持幂等。
        path.parent.mkdir(parents=True, exist_ok=True)
        # 注册限制单文件大小和历史文件数量的轮转处理器。
        handlers.append(
            RotatingFileHandler(
                path,
                maxBytes=10 * 1024 * 1024,
                backupCount=5,
                encoding="utf-8",
            )
        )

    # 统一设置根日志级别、输出格式和全部处理器。
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=handlers,
        force=True,
    )
