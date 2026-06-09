#!/usr/bin/env python3
"""DeepSWE 评估 Runner.

遍历 DeepSWE 任务集，为每个任务准备环境、运行 agent、验证结果。

用法:
    python scripts/eval_deepswe.py --task abs-module-cache-flags
    python scripts/eval_deepswe.py --limit 5
    python scripts/eval_deepswe.py --limit 10
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import tomli

# 将 src 加入路径，以便导入 ai_coding 模块
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ai_coding.config import DEFAULT_LLM_PROVIDER
from ai_coding.llm import create_lc_llm
from ai_coding.agent import LangGraphAgent
from ai_coding.logger import setup_logging
from ai_coding.tools import DEFAULT_TOOLS

from agent_common import build_system_prompt, inject_go_env


_log_file: Any = None


def set_log_file(path: Path) -> None:
    """设置日志文件，用于 background task 场景."""
    global _log_file
    _log_file = open(path, "w", encoding="utf-8")


def log(msg: str) -> None:
    """打印带时间戳的日志."""
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if _log_file is not None:
        _log_file.write(line + "\n")
        _log_file.flush()


def _rmtree_ro(path: Path) -> None:
    """删除目录树，处理 Windows 只读文件权限."""
    import stat

    def onerror(func, p, exc_info):
        if not os.access(p, os.W_OK):
            os.chmod(p, stat.S_IWUSR)
            func(p)
        else:
            raise

    shutil.rmtree(path, onerror=onerror)


def load_task_config(task_dir: Path) -> dict:
    """读取 task.toml."""
    with open(task_dir / "task.toml", "rb") as f:
        return tomli.load(f)


def load_instruction(task_dir: Path) -> str:
    """读取 instruction.md."""
    return (task_dir / "instruction.md").read_text(encoding="utf-8")


def get_cache_dir() -> Path:
    """获取本地缓存目录."""
    cache = Path.home() / ".cache" / "ai-coding" / "repos"
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def setup_repo(task_dir: Path, work_dir: Path) -> Path:
    """准备任务仓库环境.

    1. 读取 task.toml 获取 repo URL 和 base commit
    2. 优先从本地缓存复制（避免重复 clone）
    3. 缓存不存在时 git clone 到 work_dir/repo
    4. git checkout 到 base commit

    Returns:
        repo 目录路径.
    """
    config = load_task_config(task_dir)
    metadata = config["metadata"]
    repo_url = metadata["repository_url"]
    base_commit = metadata["base_commit_hash"]

    # 从 URL 提取 repo 名称作为缓存键
    repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
    cache_dir = get_cache_dir() / repo_name

    repo_dir = work_dir / "repo"
    if repo_dir.exists():
        _rmtree_ro(repo_dir)

    # 优先使用本地缓存
    if cache_dir.exists() and (cache_dir / ".git").exists():
        log(f"  Using cached repo: {cache_dir}")
        # 使用 git clone --local 代替 shutil.copytree，避免 Windows .git 权限问题
        result = subprocess.run(
            ["git", "clone", "--local", "--no-hardlinks", str(cache_dir), str(repo_dir)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            log(f"  git clone --local failed: {result.stderr}, falling back to copytree")
            shutil.copytree(cache_dir, repo_dir, ignore_dangling_symlinks=True)
    else:
        log(f"  Cloning {repo_url} ...")
        # 使用 tree:0 partial clone 加速：只下载 commit，tree/blob 按需获取
        result = subprocess.run(
            ["git", "clone", "--filter=tree:0", repo_url, str(repo_dir)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            # fallback: 普通 clone
            _rmtree_ro(repo_dir)
            result = subprocess.run(
                ["git", "clone", repo_url, str(repo_dir)],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(f"git clone failed: {result.stderr}")

        # 保存到缓存
        if not cache_dir.exists():
            shutil.copytree(repo_dir, cache_dir, ignore_dangling_symlinks=True)
            log(f"  Cached repo to: {cache_dir}")

    log(f"  Checking out {base_commit[:8]} ...")
    result = subprocess.run(
        ["git", "-C", str(repo_dir), "checkout", base_commit],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git checkout failed: {result.stderr}")

    # 配置 git 安全目录（避免在容器中出问题）
    subprocess.run(
        ["git", "config", "--global", "--add", "safe.directory", str(repo_dir)],
        capture_output=True,
    )

    return repo_dir


def _downgrade_go_mod(repo_dir: Path) -> None:
    """将 go.mod 中的 go 版本降级到本地支持的版本，避免 GOTOOLCHAIN 下载失败."""
    go_mod = repo_dir / "go.mod"
    if not go_mod.exists():
        return
    try:
        content = go_mod.read_text(encoding="utf-8")
        m = re.search(r"^go\s+(\d+)\.(\d+)$", content, re.MULTILINE)
        if m:
            major, minor = int(m.group(1)), int(m.group(2))
            # 本地 Go 1.22.4，要求 go.mod <= 1.22
            if major > 1 or (major == 1 and minor > 22):
                new_content = re.sub(r"^go\s+\d+\.\d+$", "go 1.22", content, flags=re.MULTILINE)
                go_mod.write_text(new_content, encoding="utf-8")
                log(f"  Downgraded go.mod: go {major}.{minor} -> go 1.22")
    except Exception as e:
        log(f"  Warning: failed to downgrade go.mod: {e}")


def run_agent_in_repo(
    repo_dir: Path,
    instruction: str,
    timeout: int = 300,
    max_iterations: int = 30,
) -> dict:
    """在指定目录中运行 agent.

    Args:
        repo_dir: 仓库目录，agent 的所有文件操作在此目录下进行.
        instruction: 任务指令.
        timeout: agent 最大运行时间（秒）.
        max_iterations: agent 最大 ReAct 迭代次数.

    Returns:
        包含 trajectory、final_answer、耗时等信息的字典.
    """
    # 保存当前目录，稍后恢复
    original_dir = os.getcwd()

    try:
        # 切换到仓库目录
        os.chdir(repo_dir)

        # 注入项目级便携版 Go 环境（优先于系统 Go）
        project_root = Path(__file__).parent.parent.resolve()
        inject_go_env(project_root)
        go_bin = project_root / "deep-swe" / "go" / "bin"
        if go_bin.exists():
            go_ver = subprocess.run([str(go_bin / "go.exe"), "version"], capture_output=True, text=True).stdout.strip()
            log(f"  Injected Go: {go_bin} ({go_ver})")
        # 如果 go.mod 要求的版本高于本地 Go，降级以避免 toolchain 下载失败
        _downgrade_go_mod(repo_dir)

        # 创建 agent（使用适合 SWE 任务的系统提示）
        llm = create_lc_llm(DEFAULT_LLM_PROVIDER)
        agent = LangGraphAgent(
            llm=llm,
            tools=DEFAULT_TOOLS,
            max_iterations=max_iterations,
            streaming=False,
            system_prompt=build_system_prompt(),
        )

        # 使用 run_with_trace 收集完整 trajectory
        trajectory = []
        final_answer = ""

        log(f"  Running agent (timeout={timeout}s, max_iters={max_iterations}) ...")
        start_time = time.time()

        # 在单独线程中运行 agent，以便实现超时
        import threading

        agent_result = {"done": False, "error": None, "final_answer": ""}
        trace_events = []

        def agent_worker():
            try:
                for event in agent.run_with_trace(instruction):
                    trace_events.append(event)
                    # 实时打印到 stdout（兼容 background task）
                    etype = event.get("type", "")
                    if etype == "thinking":
                        log(f"    [think] {event.get('text', '')[:200]}")
                    elif etype == "tool_call":
                        args = event.get("args", {})
                        log(f"    [tool] {event.get('name', '')}({args})")
                    elif etype == "observation":
                        text = event.get("text", "")
                        log(f"    [obs]  {text[:200].replace(chr(10), ' ')}")
                    elif etype == "assistant":
                        agent_result["final_answer"] = event.get("text", "")
                        log(f"    [answer] {event.get('text', '')[:200]}")
                    elif etype == "error":
                        agent_result["error"] = event.get("text", "")
                        log(f"    [error] {event.get('text', '')}")
                agent_result["done"] = True
            except Exception as e:
                agent_result["error"] = str(e)
                log(f"    [exception] {e}")
                agent_result["done"] = True

        worker = threading.Thread(target=agent_worker)
        worker.start()
        worker.join(timeout=timeout)

        elapsed = time.time() - start_time

        if worker.is_alive():
            # 超时，无法真正终止线程，只能标记
            log(f"  Agent timed out after {timeout}s")
            status = "timeout"
        elif agent_result["error"]:
            log(f"  Agent error: {agent_result['error']}")
            status = "error"
        else:
            log(f"  Agent finished in {elapsed:.1f}s")
            status = "done"
            final_answer = agent_result["final_answer"]

        return {
            "status": status,
            "elapsed_sec": round(elapsed, 1),
            "trajectory": trace_events,
            "final_answer": final_answer,
            "error": agent_result.get("error"),
            "message_count": len(agent.get_history()),
            "context_usage": agent.get_context_usage(),
        }

    finally:
        os.chdir(original_dir)


def save_model_patch(repo_dir: Path, output_path: Path) -> bool:
    """保存 agent 的代码修改为 model.patch.

    Returns:
        True 如果有修改，False 如果没有.
    """
    result = subprocess.run(
        ["git", "-C", str(repo_dir), "add", "-A", "--", "."],
        capture_output=True,
    )
    result = subprocess.run(
        ["git", "-C", str(repo_dir), "diff", "--cached", "--binary"],
        capture_output=True,
        text=True,
    )
    if result.stdout.strip():
        output_path.write_text(result.stdout, encoding="utf-8")
        return True
    return False


def verify_task(task_dir: Path, repo_dir: Path, work_dir: Path) -> dict:
    """运行任务验证.

    参考 Pier verifier 逻辑，在本地环境运行:
    1. 保存 agent 修改为 model.patch
    2. 重置 test.patch 涉及的文件到 base commit
    3. 应用 test.patch
    4. 运行测试

    Returns:
        验证结果字典.
    """
    log(f"  Verifying ...")

    # 保存 model.patch
    model_patch_path = work_dir / "model.patch"
    has_changes = save_model_patch(repo_dir, model_patch_path)

    if not has_changes:
        log(f"  No changes made by agent")
        return {"reward": 0, "base_pass": False, "new_pass": False, "reason": "no_changes"}

    # 读取 test.patch（使用绝对路径避免 cwd 问题）
    test_patch_path = (task_dir / "tests" / "test.patch").resolve()
    if not test_patch_path.exists():
        log(f"  No test.patch found")
        return {"reward": 0, "base_pass": False, "new_pass": False, "reason": "no_test_patch"}

    # 在验证目录中操作（复制 repo，避免污染 agent 的工作成果）
    verify_dir = work_dir / "verify"
    if verify_dir.exists():
        shutil.rmtree(verify_dir)
    shutil.copytree(repo_dir, verify_dir)

    # 解析 test.patch 中涉及的文件，重置到 HEAD 状态
    test_patch_content = test_patch_path.read_text(encoding="utf-8")
    files_to_reset = set()
    for line in test_patch_content.splitlines():
        m = re.match(r'^diff --git "?a/.+ "?b/(.+?)"?$', line)
        if m:
            files_to_reset.add(m.group(1))

    for f in sorted(files_to_reset):
        subprocess.run(
            ["git", "-C", str(verify_dir), "checkout", "HEAD", "--", f],
            capture_output=True,
        )

    # 应用 test.patch
    result = subprocess.run(
        ["git", "-C", str(verify_dir), "apply", "--whitespace=nowarn", str(test_patch_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        log(f"  Failed to apply test.patch: {result.stderr}")
        return {"reward": 0, "base_pass": False, "new_pass": False, "reason": "test_patch_apply_failed", "error": result.stderr}

    # 查找测试入口
    test_script = verify_dir / "test.sh"
    if not test_script.exists():
        log(f"  No test.sh found after applying test.patch")
        return {"reward": 0, "base_pass": False, "new_pass": False, "reason": "no_test_script"}

    # 运行 baseline 测试
    log(f"  Running baseline tests ...")
    result_base = subprocess.run(
        ["bash", str(test_script), "base"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    base_pass = result_base.returncode == 0
    log(f"  Baseline: {'PASS' if base_pass else 'FAIL'} (exit={result_base.returncode})")

    # 运行新测试
    log(f"  Running new tests ...")
    result_new = subprocess.run(
        ["bash", str(test_script), "new"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    new_pass = result_new.returncode == 0
    log(f"  New tests: {'PASS' if new_pass else 'FAIL'} (exit={result_new.returncode})")

    reward = 1 if (base_pass and new_pass) else 0

    return {
        "reward": reward,
        "base_pass": base_pass,
        "new_pass": new_pass,
        "base_output": result_base.stdout[-2000:] if result_base.stdout else "",
        "base_stderr": result_base.stderr[-1000:] if result_base.stderr else "",
        "new_output": result_new.stdout[-2000:] if result_new.stdout else "",
        "new_stderr": result_new.stderr[-1000:] if result_new.stderr else "",
    }


def run_single_task(task_dir: Path, output_base: Path, agent_timeout: int, max_iterations: int) -> dict:
    """运行单个任务的完整流程.

    Returns:
        任务结果字典.
    """
    task_id = task_dir.name
    config = load_task_config(task_dir)
    language = config["metadata"].get("language", "unknown")
    repo_url = config["metadata"]["repository_url"]

    log(f"\n{'='*60}")
    log(f"Task: {task_id} ({language})")
    log(f"Repo: {repo_url}")

    work_dir = output_base / task_id
    work_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "task_id": task_id,
        "language": language,
        "status": "pending",
        "error": None,
    }

    try:
        # 1. 准备环境
        log(f"  Setting up repo ...")
        repo_dir = setup_repo(task_dir, work_dir)
        result["repo_dir"] = str(repo_dir)

        # 2. 读取指令
        instruction = load_instruction(task_dir)

        # 3. 运行 agent
        agent_result = run_agent_in_repo(
            repo_dir,
            instruction,
            timeout=agent_timeout,
            max_iterations=max_iterations,
        )
        result.update(agent_result)

        # 4. 验证（agent 完成或超时后都验证其修改）
        if agent_result["status"] in ("done", "timeout"):
            verify_result = verify_task(task_dir, repo_dir, work_dir)
            result.update(verify_result)

        if agent_result["status"] == "done":
            result["status"] = "completed"
        elif agent_result["status"] == "timeout":
            result["status"] = "timeout"
        else:
            result["status"] = "agent_error"

    except Exception as e:
        result["status"] = "failed"
        result["error"] = traceback.format_exc()
        log(f"  ERROR: {e}")

    # 保存结果
    result_path = work_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    # 单独保存完整 trajectory（避免 result.json 过大）
    if result.get("trajectory"):
        traj_path = work_dir / "trajectory.json"
        traj_path.write_text(
            json.dumps(result["trajectory"], indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        log(f"  Trajectory saved: {traj_path} ({len(result['trajectory'])} events)")

    # 复制项目日志到结果目录
    log_file = Path("logs") / f"ai-coding-{datetime.now().strftime('%Y-%m-%d')}.log"
    if log_file.exists():
        shutil.copy2(log_file, work_dir / "agent.log")

    reward = result.get("reward", "N/A")
    log(f"  Result: {result['status']} | Reward: {reward}")

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="DeepSWE 评估 Runner")
    parser.add_argument("--tasks", default="deep-swe/tasks", help="任务目录")
    parser.add_argument("--output", default="results", help="结果输出目录")
    parser.add_argument("--task", help="运行单个任务（task_id）")
    parser.add_argument("--limit", type=int, help="限制任务数量")
    parser.add_argument("--timeout", type=int, default=300, help="Agent 超时时间（秒），默认 300")
    parser.add_argument("--max-iterations", type=int, default=50, help="Agent 最大迭代次数，默认 50")
    parser.add_argument("--sample-seed", type=int, default=None, help="随机采样种子（提供时才启用随机采样）")
    args = parser.parse_args()

    setup_logging(level="DEBUG")

    # 设置内部日志文件（background task 场景）
    log_path = Path(args.output) / "runner.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    set_log_file(log_path)

    tasks_dir = Path(args.tasks)
    if not tasks_dir.exists():
        print(f"Error: Tasks directory not found: {tasks_dir}")
        return 1

    output_base = Path(args.output)
    output_base.mkdir(parents=True, exist_ok=True)

    # 收集任务列表
    task_dirs = sorted([d for d in tasks_dir.iterdir() if d.is_dir()])

    if args.task:
        task_dirs = [d for d in task_dirs if d.name == args.task]
        if not task_dirs:
            print(f"Error: Task not found: {args.task}")
            return 1

    if args.limit:
        # 默认取前 N 个（按字母顺序），只有提供 --sample-seed 时才随机采样
        if args.sample_seed is not None:
            import random
            rng = random.Random(args.sample_seed)
            task_dirs = rng.sample(task_dirs, min(args.limit, len(task_dirs)))
            task_dirs.sort(key=lambda d: d.name)
        else:
            task_dirs = task_dirs[:args.limit]

    log(f"Total tasks to run: {len(task_dirs)}")
    log(f"Output directory: {output_base.absolute()}")
    log(f"Agent timeout: {args.timeout}s")
    log(f"Max iterations: {args.max_iterations}")

    # 运行所有任务
    all_results = []
    passed = 0
    failed = 0

    for task_dir in task_dirs:
        result = run_single_task(
            task_dir,
            output_base,
            agent_timeout=args.timeout,
            max_iterations=args.max_iterations,
        )
        all_results.append(result)

        if result.get("reward") == 1:
            passed += 1
        elif result.get("reward") == 0:
            failed += 1

    # 汇总报告
    log(f"\n{'='*60}")
    log(f"EVALUATION COMPLETE")
    log(f"{'='*60}")
    log(f"Total:   {len(all_results)}")
    log(f"Passed:  {passed}")
    log(f"Failed:  {failed}")
    log(f"Other:   {len(all_results) - passed - failed}")
    if all_results:
        log(f"Pass rate: {passed / len(all_results) * 100:.1f}%")

    # 保存汇总
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total": len(all_results),
        "passed": passed,
        "failed": failed,
        "pass_rate": round(passed / len(all_results), 4) if all_results else 0,
        "config": {
            "timeout": args.timeout,
            "max_iterations": args.max_iterations,
            "tasks_dir": str(tasks_dir),
        },
        "results": [
            {
                "task_id": r["task_id"],
                "language": r.get("language"),
                "status": r["status"],
                "reward": r.get("reward"),
                "elapsed_sec": r.get("elapsed_sec"),
                "reason": r.get("reason"),
            }
            for r in all_results
        ],
    }

    summary_path = output_base / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"Summary saved to: {summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
