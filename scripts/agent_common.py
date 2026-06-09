"""Agent 运行共享逻辑.

供 eval_deepswe.py、run_single.py、test_task.py、test_multi_task.py、
test_approval_kimi.py 等脚本复用，消除重复代码.
"""

import os
import subprocess
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
        "2. Edit：使用 edit_file 或 write_file 执行修改。\n"
        "3. Verify：修改完成后运行测试验证。\n\n"
        "绝对规则（违反会导致任务失败）：\n"
        "- 不要在探索上浪费超过 5-8 步。\n"
        "- edit_file 要求 old_string 在文件中唯一出现，增加上下文确保唯一性。\n"
        "- 修改应该最小化，只改动必要的部分。\n\n"
        "可以使用 set_todo 工具分解复杂任务，跟踪子任务进度。\n"
    )


def inject_go_env(project_root: Path) -> None:
    """注入便携版 Go 环境到 PATH（如果存在）."""
    go_bin = project_root / "deep-swe" / "go" / "bin"
    if go_bin.exists() and str(go_bin) not in os.environ.get("PATH", ""):
        os.environ["PATH"] = str(go_bin) + os.pathsep + os.environ.get("PATH", "")
        os.environ["GOTOOLCHAIN"] = "local"
        os.environ["GOPROXY"] = "https://goproxy.cn,direct"


def verify_test_task(task_dir: Path, repo_dir: Path, work_dir: Path) -> dict:
    """运行 test-tasks 风格的验证.

    1. 保存 agent 修改为 model.patch
    2. 应用 test.patch
    3. 运行生成的测试文件

    Args:
        task_dir: 原始任务目录（包含 tests/test.patch）.
        repo_dir: Agent 工作后的代码目录.
        work_dir: 临时工作目录（用于保存 model.patch）.

    Returns:
        {"reward": 0|1, "reason": str, "output": str|None, "stderr": str|None}
    """
    # 延迟导入避免循环依赖（eval_deepswe 也从本模块导入）
    from eval_deepswe import save_model_patch

    model_patch_path = work_dir / "model.patch"
    has_changes = save_model_patch(repo_dir, model_patch_path)

    if not has_changes:
        return {"reward": 0, "reason": "no_changes"}

    test_patch_path = (task_dir / "tests" / "test.patch").resolve()
    if not test_patch_path.exists():
        return {"reward": 0, "reason": "no_test_patch"}

    result = subprocess.run(
        ["git", "-C", str(repo_dir), "apply", "--whitespace=nowarn", str(test_patch_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return {"reward": 0, "reason": "test_patch_apply_failed", "stderr": result.stderr}

    test_files = list(repo_dir.glob("test_*.py"))
    if not test_files:
        return {"reward": 0, "reason": "no_test_file"}

    test_file = test_files[0]
    result = subprocess.run(
        ["python", str(test_file)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    passed = result.returncode == 0

    return {
        "reward": 1 if passed else 0,
        "reason": "passed" if passed else "test_failed",
        "output": result.stdout,
        "stderr": result.stderr,
    }
