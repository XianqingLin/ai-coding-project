"""单轮 Agent 评估脚本.

用于"人机协同"评估模式：当前对话中的 AI/Kimi Code 作为评测员，
通过多次调用本脚本驱动被测 Agent 完成多轮任务.

用法示例:
    # 第一轮：传入任务描述
    python scripts/run_eval_round.py \
        --task "请用 Python 实现一个控制台俄罗斯方块游戏" \
        --work-dir ./eval_workspace

    # 后续轮次：传入上一步 Agent 输出后，评测员决定的下一步
    python scripts/run_eval_round.py \
        --user-input "请添加旋转功能" \
        --work-dir ./eval_workspace

    # 查看当前 eval session 的历史
    python scripts/run_eval_round.py --show-history --work-dir ./eval_workspace

    # 重置 eval session（删除已有会话）
    python scripts/run_eval_round.py --reset --work-dir ./eval_workspace

每轮运行后，脚本会输出：
- Agent 的文本回复
- Agent 调用的工具列表
- 当前会话的历史摘要
- 工作目录中的文件变化（可选）
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


# 将项目 src 加入路径
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root / "src"))

from ai_coding.agent import AgentService
from ai_coding.logger import setup_logging

# 评测会话的固定名称
EVAL_SESSION_NAME = "eval-runner"


def safe_print(text: str) -> None:
    """安全打印，避免 Windows GBK 终端遇到 emoji 等特殊字符时崩溃."""
    try:
        print(text)
    except UnicodeEncodeError:
        try:
            encoded = text.encode(sys.stdout.encoding or "utf-8", errors="replace")
            sys.stdout.buffer.write(encoded + b"\n")
        except Exception:
            # 最后兜底：忽略无法编码的字符
            print(text.encode("ascii", errors="ignore").decode("ascii"))


def _list_dir_files(work_dir: Path) -> List[str]:
    """列出工作目录中的文件（用于观察 Agent 操作结果）."""
    if not work_dir.exists():
        return []
    files = []
    for p in work_dir.rglob("*"):
        if p.is_file():
            try:
                rel = p.relative_to(work_dir)
                size = p.stat().st_size
                files.append(f"{rel} ({size} bytes)")
            except Exception:
                pass
    return sorted(files)


def _format_agent_turn(service: AgentService, agent_output: str) -> str:
    """格式化 Agent 这一轮的行为."""
    tool_lines: List[str] = []

    history = service.get_history()
    recent_messages = history[-8:] if len(history) > 8 else history
    for msg in recent_messages:
        tool_calls = msg.get("tool_calls") or []
        for tc in tool_calls:
            func = tc.get("function", {})
            name = func.get("name", "")
            args = func.get("arguments", {})
            args_preview = json.dumps(args, ensure_ascii=False, default=str)
            if len(args_preview) > 200:
                args_preview = args_preview[:200] + "..."
            tool_lines.append(f"  - {name}({args_preview})")

    parts = []
    if agent_output.strip():
        parts.append(f"Agent 回复：\n{agent_output.strip()}")
    if tool_lines:
        parts.append("Agent 调用的工具：\n" + "\n".join(tool_lines))

    return "\n\n".join(parts) if parts else "（Agent 没有文本回复）"


def _get_or_create_eval_session(service: AgentService) -> Optional[str]:
    """获取或创建名为 eval-runner 的评测会话."""
    sessions = service.list_sessions()
    for s in sessions:
        if s.get("name") == EVAL_SESSION_NAME:
            sid = s.get("session_id")
            if sid:
                service.switch_session(sid)
                return sid

    # 创建新会话
    return service.create_session(name=EVAL_SESSION_NAME)


def _show_history(service: AgentService) -> None:
    """打印当前会话的简化历史."""
    history = service.get_history()
    if not history:
        safe_print("当前会话历史为空。")
        return

    safe_print(f"\n当前会话历史（共 {len(history)} 条消息）：")
    for i, msg in enumerate(history, 1):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if len(content) > 120:
            content = content[:120] + "..."
        safe_print(f"  {i}. [{role}] {content}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Agent 单轮评估脚本（人机协同模式）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例：
  # 第一轮：启动任务
  python scripts/run_eval_round.py --task "实现一个控制台俄罗斯方块" --work-dir ./eval_workspace

  # 后续轮次：传入评测员的下一步指令
  python scripts/run_eval_round.py --user-input "请添加旋转功能" --work-dir ./eval_workspace

  # 查看历史
  python scripts/run_eval_round.py --show-history --work-dir ./eval_workspace

  # 重置评测会话
  python scripts/run_eval_round.py --reset --work-dir ./eval_workspace
""",
    )
    parser.add_argument(
        "--task",
        help="任务描述。仅在第一次启动评测会话时使用。",
    )
    parser.add_argument(
        "--user-input",
        help="当前轮要发送给 Agent 的用户消息。",
    )
    parser.add_argument(
        "--work-dir",
        default=".",
        help="Agent 工作目录（默认当前目录）",
    )
    parser.add_argument(
        "--show-history",
        action="store_true",
        help="仅显示当前评测会话的历史，不运行 Agent",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="删除已有的评测会话并退出",
    )
    parser.add_argument(
        "--show-files",
        action="store_true",
        help="运行后列出工作目录中的文件",
    )
    args = parser.parse_args()

    setup_logging()

    work_dir_path = Path(args.work_dir).resolve()
    work_dir_path.mkdir(parents=True, exist_ok=True)

    # 初始化 AgentService
    service = AgentService(work_dir=str(work_dir_path), auto_approve=True)

    # 获取或创建评测会话
    eval_sid = _get_or_create_eval_session(service)
    if eval_sid is None:
        safe_print("[错误] 无法创建或切换到评测会话")
        return

    # 处理重置请求
    if args.reset:
        service.delete_session(eval_sid)
        safe_print(f"[已重置] 评测会话 {EVAL_SESSION_NAME} 已删除")
        return

    # 处理查看历史请求
    if args.show_history:
        _show_history(service)
        return

    # 确定当前轮的用户输入
    user_input: Optional[str] = None
    if args.user_input:
        user_input = args.user_input
    elif args.task:
        user_input = args.task
    else:
        safe_print("[错误] 必须提供 --task 或 --user-input 之一")
        safe_print("       使用 --show-history 查看当前会话状态")
        return

    # 记录运行前的文件列表
    files_before = _list_dir_files(work_dir_path)

    safe_print("=" * 60)
    safe_print(f"运行轮次 | 工作目录: {work_dir_path}")
    safe_print(f"User: {user_input[:200]}")
    safe_print("-" * 60)

    # 运行 Agent
    start_time = time.time()
    agent_output = service.send_message(user_input)
    elapsed = time.time() - start_time

    # 格式化输出
    turn_text = _format_agent_turn(service, agent_output)

    safe_print(turn_text)
    safe_print(f"\n[耗时: {elapsed:.2f}s]")

    # 显示文件变化
    if args.show_files:
        files_after = _list_dir_files(work_dir_path)
        new_files = [f for f in files_after if f not in files_before]
        safe_print("\n工作目录文件变化：")
        if new_files:
            safe_print("  新增/修改文件：")
            for f in new_files:
                safe_print(f"    + {f}")
        else:
            safe_print("  无文件变化")

    # 显示简要历史
    _show_history(service)

    safe_print("\n" + "=" * 60)
    safe_print("本轮结束。请输入下一步 --user-input，或 --show-history 查看历史。")


if __name__ == "__main__":
    main()
