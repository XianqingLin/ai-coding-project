"""Benchmark 任务加载器.

从 `benchmarks/tasks/<task>/task.json` 加载任务配置，并支持读取 `prompt.md`。
"""

import json
import shutil
from pathlib import Path
from typing import List

from benchmarks.spec import EvaluationSpec, TaskSpec

DEFAULT_TASK_FILE = "task.json"
DEFAULT_PROMPT_FILE = "prompt.md"


def load_task(task_dir: Path) -> TaskSpec:
    """从任务目录加载单个任务."""
    task_file = task_dir / DEFAULT_TASK_FILE
    if not task_file.exists():
        raise FileNotFoundError(f"任务配置文件不存在: {task_file}")

    with task_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    prompt = data.get("prompt", "")
    prompt_file = task_dir / DEFAULT_PROMPT_FILE
    if not prompt and prompt_file.exists():
        prompt = prompt_file.read_text(encoding="utf-8")

    evaluation = EvaluationSpec(**data["evaluation"])

    spec = TaskSpec(
        name=data["name"],
        description=data.get("description", ""),
        prompt=prompt,
        evaluation=evaluation,
        model=data.get("model"),
        max_rounds=data.get("max_rounds", 5),
        auto_approve=data.get("auto_approve", True),
        initial_files=data.get("initial_files", []),
    )
    spec.source_dir = task_dir
    return spec


def load_tasks(tasks_dir: Path) -> List[TaskSpec]:
    """加载 tasks 目录下的所有任务."""
    tasks_dir = tasks_dir.resolve()
    if not tasks_dir.exists():
        raise FileNotFoundError(f"任务目录不存在: {tasks_dir}")

    tasks: List[TaskSpec] = []
    for subdir in sorted(tasks_dir.iterdir()):
        if not subdir.is_dir():
            continue
        task_file = subdir / DEFAULT_TASK_FILE
        if task_file.exists():
            tasks.append(load_task(subdir))
    return tasks


def copy_initial_files(task_dir: Path, work_dir: Path, patterns: List[str]) -> None:
    """把任务的初始文件复制到工作目录，保持相对目录结构."""
    for pattern in patterns:
        matches = list(task_dir.glob(pattern))
        if not matches:
            raise FileNotFoundError(
                f"任务 {task_dir.name} 的 initial_files 模式未匹配: {pattern}"
            )
        for src in matches:
            rel = src.relative_to(task_dir)
            dst = work_dir / rel
            if src.is_dir():
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
