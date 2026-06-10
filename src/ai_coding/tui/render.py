"""Rich 渲染层 —— 在终端中原地输出美化内容.

支持：
- Markdown 代码块解析 + Syntax 语法高亮
- Diff 红绿着色
- 不同角色消息前缀着色
- 顶部状态栏
"""

import re
from typing import Optional

from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.rule import Rule
from rich.style import Style
from rich.syntax import Syntax
from rich.text import Text

# 全局 Console 实例（重用，避免重复初始化）
console = Console(highlight=False)

CODE_BLOCK_RE = re.compile(r"```(\w+)?\n(.*?)```", re.DOTALL)

# 角色样式映射
ROLE_PREFIX = {
    "user": ("[bold blue]>>>[/] ", Style(color="blue", bold=True)),
    "assistant": ("[bold green][Assistant][/] ", Style(color="green", bold=True)),
    "tool": ("[bold yellow][Tool][/] ", Style(color="yellow", bold=True)),
    "error": ("[bold red][Error][/] ", Style(color="red", bold=True)),
    "system": ("[bold dim][System][/] ", Style(color="bright_black", bold=True)),
    "thinking": ("[bold cyan][Thinking][/] ", Style(color="cyan", bold=True)),
}


def render_code_block(code: str, lang: str = "text") -> Panel:
    """渲染代码块为带语法高亮的 Panel."""
    try:
        syntax = Syntax(code.rstrip(), lang, theme="monokai", line_numbers=False)
    except Exception:
        syntax = Syntax(code.rstrip(), "text", theme="monokai", line_numbers=False)
    return Panel(syntax, border_style="bright_blue", padding=(0, 1))


def render_markdown(content: str) -> RenderableType:
    """解析 Markdown 代码块，混合普通文本和 Syntax Panel 输出."""
    if "```" not in content:
        return Text(content)

    parts: list[RenderableType] = []
    last_end = 0

    for match in CODE_BLOCK_RE.finditer(content):
        # 代码块前的文本
        if match.start() > last_end:
            text_part = content[last_end : match.start()].rstrip("\n")
            if text_part:
                parts.append(Text(text_part))

        # 提取代码块
        lang = match.group(1) or "text"
        code = match.group(2).rstrip()
        parts.append(render_code_block(code, lang))

        last_end = match.end()

    # 剩余文本
    if last_end < len(content):
        text_part = content[last_end:].rstrip("\n")
        if text_part:
            parts.append(Text(text_part))

    return Group(*parts) if len(parts) > 1 else (parts[0] if parts else Text(""))


def render_diff(diff_text: str) -> RenderableType:
    """渲染 Diff，+ 行绿色，- 行红色."""
    lines = diff_text.splitlines()
    result = Text()
    for line in lines:
        if line.startswith("+") and not line.startswith("+++"):
            result.append(line + "\n", style="green")
        elif line.startswith("-") and not line.startswith("---"):
            result.append(line + "\n", style="red")
        elif line.startswith("@@"):
            result.append(line + "\n", style="cyan")
        else:
            result.append(line + "\n")
    return result


def print_message(content: str, role: str = "assistant") -> None:
    """打印一条带角色前缀的消息.

    Args:
        content: 消息内容.
        role: 角色类型 (user, assistant, tool, error, system, thinking).
    """
    prefix, _ = ROLE_PREFIX.get(role, ("[dim]?[/] ", Style(color="bright_black")))
    console.print(prefix, end="")

    # 如果是 diff 内容，特殊处理
    if role == "tool" and (content.startswith("--- ") or content.startswith("diff ")):
        console.print(render_diff(content))
        return

    renderable = render_markdown(content)
    console.print(renderable)


def print_user_input(text: str) -> None:
    """打印用户输入（带 >>> 前缀）."""
    prefix, _ = ROLE_PREFIX["user"]
    console.print(f"{prefix}{text}")


def print_status_bar(work_dir: str, provider: str, status: str = "Ready") -> None:
    """打印顶部状态栏（一条分隔线 + 信息）."""
    info = f"Project: {work_dir}   Provider: {provider}   Status: {status}"
    console.print(Rule(info, style="dim"))


def print_welcome(work_dir: str, provider: str, session_name: str) -> None:
    """打印欢迎信息."""
    print_status_bar(work_dir, provider)
    console.print()
    console.print("[bold]AI Coding Assistant[/bold] — 交互式终端")
    console.print("  Commands:")
    console.print("    [dim]/help[/]     — Show help")
    console.print("    [dim]/new[/]      — Create new session")
    console.print("    [dim]/session[/]  — Session management")
    console.print("    [dim]/status[/]    — Show context usage")
    console.print("    [dim]/exit[/]     — Exit")
    console.print()


def print_help() -> None:
    """打印帮助信息."""
    console.print("[bold]Available commands:[/]")
    console.print("  [dim]/help[/]              — Show this help")
    console.print("  [dim]/new[/]               — Create a new session")
    console.print("  [dim]/session list[/]      — List all sessions")
    console.print("  [dim]/session switch ID[/] — Switch session")
    console.print("  [dim]/session rm ID[/]     — Remove session")
    console.print("  [dim]/session rename ID NAME[/]")
    console.print("  [dim]/status[/]            — Show context usage")
    console.print("  [dim]/tools[/]             — List available tools")
    console.print("  [dim]/history[/]           — Show conversation history")
    console.print("  [dim]/clear[/]             — Clear screen")
    console.print("  [dim]/verbose[/]           — Toggle verbose mode")
    console.print("  [dim]/stream[/]            — Toggle streaming mode")
    console.print("  [dim]/exit[/]              — Exit")
    console.print("  (Any other text is sent to the Agent)")
    console.print()


def print_sessions(sessions: list[dict]) -> None:
    """打印会话列表."""
    if not sessions:
        console.print("[dim]No sessions.[/]")
        return
    console.print(f"[bold]Sessions ({len(sessions)}):[/]")
    for s in sessions:
        marker = "*" if s.get("is_current") else " "
        console.print(
            f"  [{marker}] {s['session_id']}  {s['name']}  "
            f"({s.get('message_count', 0)} msgs)"
        )
    console.print()
