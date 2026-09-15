"""加载应用配置并解析其中的环境变量占位符。"""

# 启用延迟解析类型注解，避免运行时提前求值。
from __future__ import annotations

# 导入环境变量、正则、INI 解析、缓存和路径处理能力。
import os
import re
from configparser import ConfigParser
from functools import lru_cache
from pathlib import Path
from typing import Any, TypeVar

# 导入环境文件和配置模型依赖。
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field


# 定位项目根目录，作为默认配置文件和环境文件的查找基准。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# 匹配 ${NAME} 与 ${NAME:-default} 两种完整环境变量占位符。
ENV_PATTERN = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*))?\}$")
# 限定配置段泛型必须是 Pydantic 模型。
T = TypeVar("T", bound=BaseModel)


class ServerSettings(BaseModel):
    """定义 HTTP 服务的基础运行参数。"""

    # 设置服务对外展示的默认名称。
    name: str = "e-commerce-agent"
    # 默认监听所有网络接口，便于容器部署。
    host: str = "0.0.0.0"
    # 限制监听端口位于有效的 TCP 端口范围。
    port: int = Field(default=8000, ge=1, le=65535)
    # 设置服务默认日志级别。
    log_level: str = "INFO"
    # 未配置文件路径时仅输出控制台日志。
    log_file: str | None = None
    # 默认不启动尚未显式启用的外部集成。
    start_integrations: bool = False


class AppSettings(BaseModel):
    """聚合应用已知配置，并允许保留扩展配置段。"""

    # 允许数据库、JWT 等扩展配置段与核心配置共同加载。
    model_config = ConfigDict(extra="allow")

    # 未提供 server 配置段时使用安全默认值。
    server: ServerSettings = Field(default_factory=ServerSettings)


def _expand_environment(value: Any) -> Any:
    """递归展开配置值中的环境变量占位符。

    Args:
        value: 待处理的配置值，可为字典、列表或标量。

    Returns:
        展开环境变量后的配置值；非占位符值保持不变。
    """
    # 逐项处理映射并保留原有键结构。
    if isinstance(value, dict):
        # 递归展开字典中的每个配置值。
        return {key: _expand_environment(item) for key, item in value.items()}
    # 逐项处理列表并保留元素顺序。
    if isinstance(value, list):
        # 递归展开列表中的每个配置值。
        return [_expand_environment(item) for item in value]
    # 非字符串标量不包含环境变量占位符。
    if not isinstance(value, str):
        # 原样返回数字、布尔值和空值等配置。
        return value

    # 要求整个字符串符合占位符格式，避免替换普通文本片段。
    match = ENV_PATTERN.fullmatch(value)
    # 普通字符串无需展开。
    if not match:
        # 保留未匹配占位符的原始字符串。
        return value

    # 分离环境变量名称和可选默认值。
    variable, default = match.groups()
    # 优先读取环境变量，未设置时使用占位符默认值或空字符串。
    return os.getenv(variable, default if default is not None else "")


@lru_cache(maxsize=1)
def get_raw_config() -> dict[str, Any]:
    """读取配置文件并展开其中的环境变量。

    Returns:
        展开环境变量后的原始配置字典。

    Raises:
        FileNotFoundError: 配置文件不存在。
        configparser.Error: INI 配置文件格式不正确。
    """
    # 加载项目环境文件，但不覆盖进程中已经设置的变量。
    load_dotenv(PROJECT_ROOT / ".env")
    # 允许通过环境变量指定配置文件路径。
    configured_path = os.getenv("AGENT_CENTER_CONFIG")
    # 未指定路径时回退到项目根目录下的默认 INI 文件。
    config_path = Path(configured_path) if configured_path else PROJECT_ROOT / "application.ini"
    # 在读取前验证配置文件确实存在。
    if not config_path.is_file():
        # 提供包含实际路径的错误，便于定位部署配置问题。
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # 关闭 ConfigParser 的内置插值，保留自定义环境变量占位符。
    parser = ConfigParser(interpolation=None)
    # 以 UTF-8 文本模式打开配置文件并确保句柄自动关闭。
    with config_path.open("r", encoding="utf-8") as stream:
        # 交由标准 INI 解析器读取并校验文件结构。
        parser.read_file(stream)
    # 将所有配置段转换为普通字典，便于后续模型校验。
    data = {section: dict(parser.items(section, raw=True)) for section in parser.sections()}
    # 在返回前统一展开所有环境变量占位符。
    return _expand_environment(data)


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """获取经过校验的应用配置。

    Returns:
        缓存的应用配置对象。

    Raises:
        pydantic.ValidationError: 配置内容不符合应用配置模型。
    """
    # 将原始配置校验并转换为强类型应用配置。
    return AppSettings.model_validate(get_raw_config())


def get_section(name: str, model: type[T]) -> T:
    """读取并校验指定名称的配置段。

    Args:
        name: 配置段名称。
        model: 用于校验配置段的 Pydantic 模型类型。

    Returns:
        校验后的配置模型实例。

    Raises:
        RuntimeError: 指定配置段不存在或未启用。
        pydantic.ValidationError: 配置段内容不符合指定模型。
    """
    # 从完整配置中按名称查找目标配置段。
    raw = get_raw_config().get(name)
    # 缺少配置段时明确阻止依赖该能力的组件启动。
    if raw is None:
        # 使用配置段名称构造可诊断的错误信息。
        raise RuntimeError(f"Configuration section is not enabled: {name}")
    # 使用调用方指定的模型完成配置校验和类型转换。
    return model.model_validate(raw)


def clear_settings_cache() -> None:
    """清除原始配置和应用配置缓存。

    Returns:
        None.
    """
    # 清除强类型应用配置缓存，使后续调用重新加载配置。
    get_settings.cache_clear()
    # 清除原始配置缓存，使环境变量或配置文件变更生效。
    get_raw_config.cache_clear()
