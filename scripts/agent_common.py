"""Agent 运行共享逻辑.

供 eval_deepswe.py、run_single.py、test_task.py 等脚本复用，
消除重复的系统提示构建和 Go 环境注入代码.
"""

import os
from pathlib import Path


def build_system_prompt(prompt_path: str | None = None) -> str:
    """构建 SWE 任务的默认系统提示.

    如果提供了自定义提示文件路径，读取其内容作为完整系统提示。
    """
    if prompt_path:
        p = Path(prompt_path)
        if p.exists():
            return p.read_text(encoding="utf-8")

    return (
        "你是一个软件工程 agent，专门负责修改代码来完成给定的开发任务。\n"
        "当前你位于一个代码仓库的根目录中。\n\n"
        "你的工作流程（严格按此顺序执行）：\n"
        "1. 探索：使用 read_file、grep、list_dir 理解代码库。"
        "读 3-5 个关键文件后，你就必须停止探索。\n"
        "2. Plan：调用 plan 工具提交修改计划。这是进入修改阶段的唯一方式。\n"
        "3. Edit：调用 plan 后的下一步**必须**是 str_replace_file 或 write_file。"
        "不允许在 plan 后继续 read_file/grep/list_dir。\n"
        "4. Verify：修改完成后运行测试验证。\n\n"
        "绝对规则（违反会导致任务失败）：\n"
        "- 不调用 plan 就无法开始修改。\n"
        "- 调用 plan 后必须立即 edit，不能在 plan 后继续探索。\n"
        "- 不要在探索上浪费超过 5-8 步。\n"
        "- str_replace_file 要求 old_string 在文件中唯一出现，增加上下文确保唯一性。\n"
        "- 修改应该最小化，只改动必要的部分。\n"
        "- plan 不需要完美，提交初步方案即可，执行中可以调整。\n\n"
        "可以使用 set_todo 工具分解复杂任务，跟踪子任务进度。\n"
    )


def inject_go_env(project_root: Path) -> None:
    """注入便携版 Go 环境到 PATH（如果存在）."""
    go_bin = project_root / "deep-swe" / "go" / "bin"
    if go_bin.exists() and str(go_bin) not in os.environ.get("PATH", ""):
        os.environ["PATH"] = str(go_bin) + os.pathsep + os.environ.get("PATH", "")
        os.environ["GOTOOLCHAIN"] = "local"
        os.environ["GOPROXY"] = "https://goproxy.cn,direct"
