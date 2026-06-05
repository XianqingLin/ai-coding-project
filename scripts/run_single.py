#!/usr/bin/env python3
"""轻量级单任务运行器.

用于快速验证单个 DeepSWE 任务，支持交互式单步调试、自定义 prompt、
trajectory 对比.

用法:
    # 正常模式运行单个任务
    python scripts/run_single.py --task abs-module-cache-flags

    # 单步调试模式（每轮暂停，按 Enter 继续）
    python scripts/run_single.py --task abs-module-cache-flags --step

    # 使用自定义系统提示
    python scripts/run_single.py --task abs-module-cache-flags --prompt prompts/aggressive.txt

    # 缩短迭代次数快速测试
    python scripts/run_single.py --task abs-module-cache-flags --max-iters 10 --timeout 60

    # 对比两次运行 trajectory
    python scripts/run_single.py --task abs-module-cache-flags --compare results/single/abs-module-cache-flags-v1
"""

import argparse
import json
import os
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

# Windows 终端兼容：避免 UnicodeEncodeError
if os.name == "nt":
    console = Console(legacy_windows=False)
else:
    console = Console()

# 将 src 加入路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ai_coding.config import DEFAULT_LLM_PROVIDER
from ai_coding.llm import create_lc_llm
from ai_coding.langgraph_agent import LangGraphAgent
from ai_coding.logger import setup_logging
from ai_coding.tools import DEFAULT_TOOLS

# 从 eval_deepswe 复用核心函数
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from eval_deepswe import (
    _rmtree_ro,
    load_instruction,
    load_task_config,
    save_model_patch,
    setup_repo,
    verify_task,
)


def print_event(event: Dict[str, Any], step_mode: bool = False) -> None:
    """打印单个轨迹事件（彩色格式化）."""
    etype = event.get("type", "")
    step = event.get("step", "?")
    ts = event.get("timestamp", 0)

    if etype == "thinking":
        text = event.get("text", "")
        preview = text[:300].replace("\n", " ")
        if len(text) > 300:
            preview += "..."
        console.print(
            f"[bold cyan]  Step {step}[/bold cyan] [dim]({ts:.1f}s)[/dim] [yellow][THINK][/yellow]"
        )
        console.print(f"     {preview}")

    elif etype == "tool_call":
        name = event.get("name", "")
        args = event.get("args", {})
        # 简化参数显示
        args_str = json.dumps(args, ensure_ascii=False, default=str)
        if len(args_str) > 200:
            args_str = args_str[:200] + "..."
        console.print(
            f"[bold cyan]  Step {step}[/bold cyan] [dim]({ts:.1f}s)[/dim] [green][TOOL][/green] [bold]{name}[/bold]"
        )
        console.print(f"     {args_str}")

    elif etype == "observation":
        text = event.get("text", "")
        preview = text[:300].replace("\n", " ")
        if len(text) > 300:
            preview += "..."
        console.print(
            f"[bold cyan]  Step {step}[/bold cyan] [dim]({ts:.1f}s)[/dim] [blue][OBS][/blue]"
        )
        console.print(f"     {preview}")

    elif etype == "assistant":
        text = event.get("text", "")
        preview = text[:300].replace("\n", " ")
        if len(text) > 300:
            preview += "..."
        console.print(
            f"[bold cyan]  Step {step}[/bold cyan] [dim]({ts:.1f}s)[/dim] [magenta][ANSWER][/magenta]"
        )
        console.print(f"     {preview}")

    elif etype == "error":
        console.print(
            f"[bold red]  Step {step}[/bold red] [dim]({ts:.1f}s)[/dim] [red][ERROR][/red] {event.get('text', '')}"
        )

    if step_mode:
        console.print("  [dim]─" * 40 + "[/dim]")


def run_agent_step_mode(
    repo_dir: Path,
    instruction: str,
    system_prompt: str,
    max_iterations: int,
    provider: str,
) -> Dict[str, Any]:
    """单步模式：在主线程运行，每 yield 一个 event 暂停等用户确认.

    不设置 timeout，由用户控制节奏.
    """
    original_dir = os.getcwd()
    try:
        os.chdir(repo_dir)

        # 注入便携版 Go 环境
        go_bin = PROJECT_ROOT / "deep-swe" / "go" / "bin"
        if go_bin.exists() and str(go_bin) not in os.environ.get("PATH", ""):
            os.environ["PATH"] = str(go_bin) + os.pathsep + os.environ.get("PATH", "")
            os.environ["GOTOOLCHAIN"] = "local"
            os.environ["GOPROXY"] = "https://goproxy.cn,direct"

        llm = create_lc_llm(provider)
        agent = LangGraphAgent(
            llm=llm,
            tools=DEFAULT_TOOLS,
            max_iterations=max_iterations,
            streaming=False,
            system_prompt=system_prompt,
        )

        trajectory: List[Dict[str, Any]] = []
        final_answer = ""
        error_msg = None

        console.print("\n[bold green]> Agent 开始运行（Step 模式）[/bold green]")
        console.print("[dim]  每步暂停，按 Enter 继续，输入 'q' 退出，输入 'c' 取消暂停[/dim]\n")

        start_time = time.time()
        auto_continue = False

        try:
            for event in agent.run_with_trace(instruction):
                trajectory.append(event)
                print_event(event, step_mode=True)

                if event.get("type") == "assistant":
                    final_answer = event.get("text", "")

                if not auto_continue:
                    try:
                        user_input = console.input(
                            "  [bold yellow]⏸ 按 Enter 继续 / [c]取消暂停 / [q]退出:[/bold yellow] "
                        ).strip().lower()
                    except (EOFError, KeyboardInterrupt):
                        user_input = "q"

                    if user_input == "q":
                        console.print("  [red]用户中断[/red]")
                        break
                    elif user_input == "c":
                        auto_continue = True
                        console.print("  [dim]取消暂停，后续自动继续...[/dim]")
        except Exception as e:
            error_msg = str(e)
            console.print(f"[red]  Agent 异常: {e}[/red]")

        elapsed = time.time() - start_time

        console.print(f"\n[bold green]> Agent 结束[/bold green] [dim]({elapsed:.1f}s, {len(trajectory)} events)[/dim]")

        return {
            "status": "done" if not error_msg else "error",
            "elapsed_sec": round(elapsed, 1),
            "trajectory": trajectory,
            "final_answer": final_answer,
            "error": error_msg,
            "message_count": len(agent.get_history()),
            "context_usage": agent.get_context_usage(),
        }

    finally:
        os.chdir(original_dir)


def run_agent_normal(
    repo_dir: Path,
    instruction: str,
    system_prompt: str,
    max_iterations: int,
    timeout: int,
    provider: str,
) -> Dict[str, Any]:
    """正常模式：在线程中运行，支持 timeout.

    与 eval_deepswe.py 的 run_agent_in_repo 基本一致，但使用 rich 彩色输出.
    """
    original_dir = os.getcwd()
    try:
        os.chdir(repo_dir)

        # 注入便携版 Go 环境
        go_bin = PROJECT_ROOT / "deep-swe" / "go" / "bin"
        if go_bin.exists() and str(go_bin) not in os.environ.get("PATH", ""):
            os.environ["PATH"] = str(go_bin) + os.pathsep + os.environ.get("PATH", "")
            os.environ["GOTOOLCHAIN"] = "local"
            os.environ["GOPROXY"] = "https://goproxy.cn,direct"

        llm = create_lc_llm(provider)
        agent = LangGraphAgent(
            llm=llm,
            tools=DEFAULT_TOOLS,
            max_iterations=max_iterations,
            streaming=False,
            system_prompt=system_prompt,
        )

        trajectory: List[Dict[str, Any]] = []
        agent_result = {"done": False, "error": None, "final_answer": ""}

        console.print("\n[bold green]> Agent 开始运行[/bold green]")
        start_time = time.time()

        def agent_worker():
            try:
                for event in agent.run_with_trace(instruction):
                    trajectory.append(event)
                    print_event(event, step_mode=False)
                    if event.get("type") == "assistant":
                        agent_result["final_answer"] = event.get("text", "")
                agent_result["done"] = True
            except Exception as e:
                agent_result["error"] = str(e)
                console.print(f"[red]  Agent 异常: {e}[/red]")
                agent_result["done"] = True

        worker = threading.Thread(target=agent_worker)
        worker.start()
        worker.join(timeout=timeout)

        elapsed = time.time() - start_time

        if worker.is_alive():
            console.print(f"\n[bold red]> Agent 超时 ({timeout}s)[/bold red]")
            status = "timeout"
        elif agent_result["error"]:
            console.print(f"\n[bold red]> Agent 错误[/bold red]")
            status = "error"
        else:
            console.print(f"\n[bold green]> Agent 完成[/bold green] [dim]({elapsed:.1f}s)[/dim]")
            status = "done"

        return {
            "status": status,
            "elapsed_sec": round(elapsed, 1),
            "trajectory": trajectory,
            "final_answer": agent_result["final_answer"],
            "error": agent_result.get("error"),
            "message_count": len(agent.get_history()),
            "context_usage": agent.get_context_usage(),
        }

    finally:
        os.chdir(original_dir)


def build_system_prompt(prompt_path: str | None) -> str:
    """构建系统提示.

    如果提供了 prompt 文件路径，读取其内容作为完整系统提示。
    否则使用默认的 SWE 系统提示.
    """
    if prompt_path:
        p = Path(prompt_path)
        if p.exists():
            return p.read_text(encoding="utf-8")
        else:
            console.print(f"[yellow]警告: 提示文件不存在: {prompt_path}，使用默认提示[/yellow]")

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


def compare_trajectories(current: List[Dict], previous_dir: Path) -> None:
    """对比当前运行与之前运行的 trajectory 差异."""
    traj_path = previous_dir / "trajectory.json"
    if not traj_path.exists():
        console.print(f"[red]找不到之前的 trajectory: {traj_path}[/red]")
        return

    previous = json.loads(traj_path.read_text(encoding="utf-8"))

    console.print(Rule("[bold]Trajectory 对比"))
    console.print(f"  当前:   {len(current)} events")
    console.print(f"  之前:   {len(previous)} events")

    # 对比工具调用序列
    current_tools = [
        (e.get("step"), e.get("name"), json.dumps(e.get("args", {}), ensure_ascii=False, default=str)[:80])
        for e in current if e.get("type") == "tool_call"
    ]
    previous_tools = [
        (e.get("step"), e.get("name"), json.dumps(e.get("args", {}), ensure_ascii=False, default=str)[:80])
        for e in previous if e.get("type") == "tool_call"
    ]

    console.print(f"\n  [bold]工具调用对比:[/bold]")
    max_len = max(len(current_tools), len(previous_tools))
    diff_count = 0
    for i in range(max_len):
        cur = current_tools[i] if i < len(current_tools) else None
        prev = previous_tools[i] if i < len(previous_tools) else None

        if cur != prev:
            diff_count += 1
            console.print(f"  [red]  #{i+1} DIFF[/red]")
            if cur:
                console.print(f"    当前: step={cur[0]} {cur[1]} {cur[2]}")
            if prev:
                console.print(f"    之前: step={prev[0]} {prev[1]} {prev[2]}")

    if diff_count == 0:
        console.print("  [green]  工具调用序列完全一致[/green]")
    else:
        console.print(f"\n  [yellow]共 {diff_count} 处差异[/yellow]")

    # 对比最终状态
    current_has_plan = any(
        e.get("type") == "tool_call" and e.get("name") == "plan"
        for e in current
    )
    previous_has_plan = any(
        e.get("type") == "tool_call" and e.get("name") == "plan"
        for e in previous
    )
    console.print(f"\n  [bold]是否调用 plan:[/bold]")
    console.print(f"    当前: {'[green]是[/green]' if current_has_plan else '[red]否[/red]'}")
    console.print(f"    之前: {'[green]是[/green]' if previous_has_plan else '[red]否[/red]'}")

    current_has_edit = any(
        e.get("type") == "tool_call" and e.get("name") in ("str_replace_file", "write_file")
        for e in current
    )
    previous_has_edit = any(
        e.get("type") == "tool_call" and e.get("name") in ("str_replace_file", "write_file")
        for e in previous
    )
    console.print(f"\n  [bold]是否执行编辑:[/bold]")
    console.print(f"    当前: {'[green]是[/green]' if current_has_edit else '[red]否[/red]'}")
    console.print(f"    之前: {'[green]是[/green]' if previous_has_edit else '[red]否[/red]'}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="轻量级单任务运行器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/run_single.py --task abs-module-cache-flags
  python scripts/run_single.py --task abs-module-cache-flags --step
  python scripts/run_single.py --task abs-module-cache-flags --prompt prompts/v2.txt --max-iters 15
  python scripts/run_single.py --task abs-module-cache-flags --compare results/single/abs-module-cache-flags-v1
        """,
    )
    parser.add_argument("--task", required=True, help="任务 ID（如 abs-module-cache-flags）")
    parser.add_argument("--tasks-dir", default="deep-swe/tasks", help="任务目录")
    parser.add_argument("--output", default="results/single", help="结果输出目录")
    parser.add_argument("--timeout", type=int, default=300, help="Agent 超时时间（秒），默认 300")
    parser.add_argument("--max-iters", type=int, default=50, help="Agent 最大迭代次数，默认 50")
    parser.add_argument("--prompt", help="自定义系统提示文件路径")
    parser.add_argument("--step", action="store_true", help="单步模式，每轮暂停等确认")
    parser.add_argument("--provider", default=DEFAULT_LLM_PROVIDER, help=f"LLM provider（默认: {DEFAULT_LLM_PROVIDER}）")
    parser.add_argument("--no-verify", action="store_true", help="跳过验证")
    parser.add_argument("--compare", help="对比之前运行的结果目录")
    parser.add_argument("--clean", action="store_true", help="运行前清理之前的输出目录")
    args = parser.parse_args()

    setup_logging(level="DEBUG")

    task_dir = Path(args.tasks_dir) / args.task
    if not task_dir.exists():
        console.print(f"[red]错误: 任务目录不存在: {task_dir}[/red]")
        return 1

    # 输出目录命名：如果已存在则加序号
    output_base = Path(args.output)
    work_dir = output_base / args.task
    if work_dir.exists() and args.clean:
        console.print(f"[dim]清理旧目录: {work_dir}[/dim]")
        _rmtree_ro(work_dir)

    # 如果目录已存在且没 --clean，创建带序号的目录
    counter = 1
    original_work_dir = work_dir
    while work_dir.exists():
        work_dir = Path(f"{original_work_dir}-v{counter}")
        counter += 1

    work_dir.mkdir(parents=True, exist_ok=True)

    console.print(Rule(f"[bold blue]Task: {args.task}[/bold blue]"))

    # 读取任务信息
    config = load_task_config(task_dir)
    metadata = config["metadata"]
    console.print(f"  语言:    {metadata.get('language', 'unknown')}")
    console.print(f"  仓库:    {metadata['repository_url']}")
    console.print(f"  Commit:  {metadata['base_commit_hash'][:8]}")
    console.print(f"  输出:    {work_dir}")
    console.print(f"  模式:    {'[yellow]单步调试[/yellow]' if args.step else '正常'}")
    console.print(f"  Provider: {args.provider}")
    console.print(f"  Timeout: {args.timeout}s | Max iters: {args.max_iters}")

    result = {
        "task_id": args.task,
        "language": metadata.get("language", "unknown"),
        "status": "pending",
        "error": None,
    }

    try:
        # 1. 准备环境
        console.print(f"\n[bold]1. 准备仓库环境[/bold]")
        repo_dir = setup_repo(task_dir, work_dir)
        result["repo_dir"] = str(repo_dir)

        # 2. 读取指令
        instruction = load_instruction(task_dir)
        console.print(f"  指令长度: {len(instruction)} chars")

        # 3. 构建系统提示
        system_prompt = build_system_prompt(args.prompt)
        if args.prompt:
            console.print(f"  使用自定义提示: {args.prompt}")

        # 4. 运行 Agent
        console.print(f"\n[bold]2. 运行 Agent[/bold]")
        if args.step:
            agent_result = run_agent_step_mode(
                repo_dir,
                instruction,
                system_prompt,
                args.max_iters,
                args.provider,
            )
        else:
            agent_result = run_agent_normal(
                repo_dir,
                instruction,
                system_prompt,
                args.max_iters,
                args.timeout,
                args.provider,
            )

        result.update(agent_result)

        # 5. 验证
        if not args.no_verify and agent_result["status"] in ("done", "timeout"):
            console.print(f"\n[bold]3. 验证结果[/bold]")
            verify_result = verify_task(task_dir, repo_dir, work_dir)
            result.update(verify_result)

            reward = verify_result.get("reward", "N/A")
            if reward == 1:
                console.print(f"  [bold green]OK 验证通过 (reward=1)[/bold green]")
            elif reward == 0:
                reason = verify_result.get("reason", "unknown")
                console.print(f"  [bold red]FAIL 验证失败 (reward=0, reason={reason})[/bold red]")
            else:
                console.print(f"  验证结果: reward={reward}")

        # 确定最终状态
        if result.get("reward") == 1:
            result["status"] = "passed"
        elif result.get("reward") == 0:
            result["status"] = "failed"
        elif agent_result["status"] == "timeout":
            result["status"] = "timeout"
        elif agent_result["status"] == "error":
            result["status"] = "agent_error"
        else:
            result["status"] = "completed"

    except Exception as e:
        result["status"] = "failed"
        result["error"] = traceback.format_exc()
        console.print(f"[red]运行失败: {e}[/red]")

    # 保存结果
    result_path = work_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    # 保存 trajectory
    if result.get("trajectory"):
        traj_path = work_dir / "trajectory.json"
        traj_path.write_text(
            json.dumps(result["trajectory"], indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        console.print(f"\n  Trajectory: {traj_path} ({len(result['trajectory'])} events)")

    # 对比
    if args.compare:
        compare_dir = Path(args.compare)
        if compare_dir.exists():
            compare_trajectories(result.get("trajectory", []), compare_dir)
        else:
            console.print(f"[yellow]对比目录不存在: {compare_dir}[/yellow]")

    # 最终摘要
    console.print(Rule("[bold]运行摘要"))
    console.print(f"  任务:    {args.task}")
    console.print(f"  状态:    {result['status']}")
    console.print(f"  耗时:    {result.get('elapsed_sec', 'N/A')}s")
    console.print(f"  消息数:  {result.get('message_count', 'N/A')}")
    console.print(f"  Context: {result.get('context_usage', {}).get('usage_percent', 'N/A')}%")
    if "reward" in result:
        console.print(f"  Reward:  {result['reward']}")
    console.print(f"  输出:    {work_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
