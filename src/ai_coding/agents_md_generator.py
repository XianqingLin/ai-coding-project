"""AGENTS.md 生成器.

分析项目结构并生成项目级稳定记忆文档，供 AI Agent 后续会话读取.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from ai_coding.llm import create_lc_llm
from ai_coding.logger import get_logger

logger = get_logger(__name__)

DEFAULT_READ_LIMIT = 2000


def _read_first_n_chars(path: Path, limit: int = DEFAULT_READ_LIMIT) -> str:
    """读取文件前 N 个字符，失败返回空字符串."""
    try:
        text = path.read_text(encoding="utf-8")
        return text[:limit]
    except Exception as e:
        logger.warning(f"读取文件失败 {path}: {e}")
        return ""


def _list_directory_tree(root: Path, max_depth: int = 2) -> List[str]:
    """列出目录树，控制深度避免过大."""
    entries: List[str] = []

    def walk(path: Path, depth: int) -> None:
        if depth > max_depth:
            return
        for p in sorted(path.iterdir()):
            name = p.name
            if name.startswith("__") or name.startswith("."):
                continue
            rel = p.relative_to(root)
            entries.append("  " * depth + str(rel))
            if p.is_dir():
                walk(p, depth + 1)

    walk(root, 0)
    return entries


def _collect_project_info(work_dir: Path) -> Dict[str, Any]:
    """收集项目关键信息."""
    info: Dict[str, Any] = {
        "work_dir": str(work_dir),
        "top_level": [],
        "readme": "",
        "config_files": {},
        "source_tree": [],
        "existing_agents_md": "",
    }

    # 顶层文件/目录
    for p in sorted(work_dir.iterdir()):
        if p.name.startswith("."):
            continue
        info["top_level"].append(p.name)

    # README
    readme_path = work_dir / "README.md"
    if readme_path.exists():
        info["readme"] = _read_first_n_chars(readme_path)

    # 关键配置文件
    config_candidates = [
        "pyproject.toml",
        "package.json",
        "setup.py",
        "setup.cfg",
        "requirements.txt",
        "Makefile",
    ]
    for name in config_candidates:
        config_path = work_dir / name
        if config_path.exists():
            info["config_files"][name] = _read_first_n_chars(config_path)

    # 源码目录结构
    src_dir = work_dir / "src"
    if src_dir.exists():
        info["source_tree"] = _list_directory_tree(src_dir, max_depth=2)

    # 现有 AGENTS.md
    agents_md_path = work_dir / "AGENTS.md"
    if agents_md_path.exists():
        info["existing_agents_md"] = _read_first_n_chars(agents_md_path, limit=10000)

    return info


def _build_prompt(info: Dict[str, Any]) -> str:
    """构造生成 AGENTS.md 的 prompt."""
    sections: List[str] = []
    sections.append(
        "请根据以下项目信息，生成一份简洁、稳定的 AGENTS.md 文件。"
        "AGENTS.md 用于向 AI Agent 传达项目级长期记忆，"
        "包括技术栈、常用命令、代码规范和架构要点。"
        "输出应为纯 Markdown，不要包含解释性文字。"
    )

    sections.append(f"\n项目目录: {info['work_dir']}")
    sections.append(f"顶层文件/目录: {', '.join(info['top_level'])}")

    if info["readme"]:
        sections.append(f"\nREADME 节选:\n```\n{info['readme']}\n```")

    for name, content in info["config_files"].items():
        sections.append(f"\n{name} 节选:\n```\n{content}\n```")

    if info["source_tree"]:
        tree_text = "\n".join(info["source_tree"])
        sections.append(f"\n源码目录结构:\n```\n{tree_text}\n```")

    if info["existing_agents_md"]:
        sections.append(
            f"\n现有 AGENTS.md 内容:\n```markdown\n"
            f"{info['existing_agents_md']}\n```"
        )
        sections.append(
            "\n请在现有内容基础上更新，保留仍然正确的部分，补充缺失的部分。"
        )

    sections.append("\n建议包含以下章节：")
    sections.append("- 项目概述")
    sections.append("- 技术栈")
    sections.append("- 常用命令（测试、检查、运行）")
    sections.append("- 代码规范")
    sections.append("- 架构要点")
    sections.append("- 重要文件")
    sections.append("- Agent 协作提示")

    return "\n".join(sections)


def generate_agents_md(
    work_dir: Path,
    llm: Optional[Any] = None,
) -> str:
    """生成 AGENTS.md 内容.

    Args:
        work_dir: 项目根目录.
        llm: 可选 LLM 实例，未提供时使用默认配置创建.

    Returns:
        生成的 Markdown 文本.

    Raises:
        RuntimeError: LLM 调用失败时抛出.
    """
    info = _collect_project_info(work_dir)

    if llm is None:
        llm = create_lc_llm()

    system_prompt = (
        "你是一名技术文档专家。你的任务是根据项目信息生成 AGENTS.md，"
        "这份文档将作为 AI Agent 的项目级长期记忆。文档应简洁、准确、"
        "聚焦于对后续开发协作有用的稳定信息。"
    )
    user_prompt = _build_prompt(info)

    try:
        response = llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )
        raw_content = (
            response.content if hasattr(response, "content") else str(response)
        )
        if isinstance(raw_content, list):
            content = "\n".join(str(item) for item in raw_content)
        else:
            content = str(raw_content)
    except Exception as e:
        raise RuntimeError(f"调用 LLM 生成 AGENTS.md 失败: {e}") from e

    return content.strip()


def write_agents_md(
    work_dir: Path,
    content: str,
    overwrite: bool = False,
) -> bool:
    """写入 AGENTS.md 文件.

    Args:
        work_dir: 项目根目录.
        content: 要写入的内容.
        overwrite: 是否覆盖已存在的文件.

    Returns:
        是否实际写入了文件（已存在且不覆盖时返回 False）.
    """
    path = work_dir / "AGENTS.md"

    if path.exists() and not overwrite:
        return False

    path.write_text(content, encoding="utf-8")
    return True
