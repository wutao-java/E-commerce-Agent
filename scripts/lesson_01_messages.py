"""第 01 课：运行一次本地模型消息验证。"""

from model import (
    build_messages,
    call_chat_model,
    extract_assistant_message,
)


def main() -> int:
    """执行一次本地模型消息验证。"""

    try:
        response = call_chat_model(build_messages())
        assistant_message = extract_assistant_message(response)
    except RuntimeError as exc:
        print(f"模型调用未完成：{exc}")
        return 1

    print("小哲电商客服 Agent 回复：")
    print(assistant_message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())