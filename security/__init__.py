"""对外暴露 JWT 验签接口。"""

# 导入安全包提供的公共 JWT 操作。
from .jwt import decode_token

# 明确限制安全包的公共导出符号。
__all__ = ["decode_token"]
