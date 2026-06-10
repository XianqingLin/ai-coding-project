"""协作类工具.

提供 AskUserQuestion（向用户提问）和 Agent（委派子 Agent）两个工具.
"""

from typing import Any, Callable, List, Optional

from ai_coding.agent.sub_agent_manager import SubAgentManager
from ai_coding.tools.base import Tool, ToolParameter


class AskUserQuestionTool(Tool):
    """以结构化多选形式向用户提问，用于消歧或选择方案."""

    name = "ask_user_question"
    requires_approval = False
    description = (
        "当你需要消歧、让用户选择方案或收集用户偏好时，使用此工具向用户提问。\n"
        "支持单选或多选，用户可以输入选项编号或直接输入自定义回答。"
    )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter("question", "string", "要向用户提出的问题"),
            ToolParameter(
                "options",
                "array",
                "选项列表，每项为包含 'label'(str) 和可选 'description'(str) 的字典",
                required=False,
                default=[],
            ),
            ToolParameter(
                "multi_select",
                "boolean",
                "是否允许多选，默认 false",
                required=False,
                default=False,
            ),
        ]

    def execute(
        self,
        question: str = "",
        options: List[Any] = None,
        multi_select: bool = False,
    ) -> str:
        if not question:
            return "[错误] question 参数不能为空"

        options = options or []
        validated = []
        for i, opt in enumerate(options):
            if isinstance(opt, dict) and "label" in opt:
                validated.append({
                    "label": str(opt["label"]),
                    "description": str(opt.get("description", "")),
                })

        print(f"\n[AskUserQuestion] {question}")
        print("-" * 50)
        for i, opt in enumerate(validated, 1):
            desc = opt["description"]
            if desc:
                print(f"  {i}) {opt['label']} - {desc}")
            else:
                print(f"  {i}) {opt['label']}")
        print("  0) 其他（自定义输入）")
        print("-" * 50)

        while True:
            try:
                if multi_select:
                    choice = input("请选择（多选用逗号分隔，如 1,3）: ").strip()
                else:
                    choice = input("请选择: ").strip()
            except (EOFError, KeyboardInterrupt):
                return "[系统] 用户取消输入。"

            if not choice:
                print("输入不能为空，请重新选择。")
                continue

            # 解析选择
            if multi_select:
                parts = [p.strip() for p in choice.split(",")]
                selected_labels = []
                invalid = []
                for p in parts:
                    try:
                        idx = int(p)
                        if idx == 0:
                            selected_labels.append("其他")
                        elif 1 <= idx <= len(validated):
                            selected_labels.append(validated[idx - 1]["label"])
                        else:
                            invalid.append(p)
                    except ValueError:
                        invalid.append(p)
                if invalid:
                    print(f"无效选项: {', '.join(invalid)}，请重新选择。")
                    continue
                return f"[用户选择] {', '.join(selected_labels)}"
            else:
                try:
                    idx = int(choice)
                    if idx == 0:
                        custom = input("请输入你的回答: ").strip()
                        return f"[用户回答] {custom}"
                    elif 1 <= idx <= len(validated):
                        label = validated[idx - 1]["label"]
                        return f"[用户选择] {label}"
                    else:
                        print("无效选项，请重新选择。")
                        continue
                except ValueError:
                    # 用户直接输入了文本
                    return f"[用户回答] {choice}"


class AgentTool(Tool):
    """将子任务委托给子 Agent 执行."""

    name = "agent"
    requires_approval = True
    description = (
        "将子任务委派给子 Agent 执行。子 Agent 拥有独立的上下文和工具集，"
        "不会污染主 Agent 的对话历史。\n"
        "内置三种子 Agent：\n"
        "- coder（默认）：通用软件工程助手，可读写文件、执行命令\n"
        "- explore：只读探索，适合快速梳理代码库\n"
        "- plan：只读+规划，专注于架构设计，不执行命令\n\n"
        "前台模式下父 Agent 等待子 Agent 完成再继续；"
        "后台模式立即返回任务 ID，完成后结果自动回到主 Agent。"
    )

    def __init__(self) -> None:
        super().__init__()
        self._llm: Any = None
        self._llm_factory: Optional[Callable[[], Any]] = None
        from ai_coding.agent.sub_agent_manager import SubAgentManager
        self._manager = SubAgentManager()

    def set_llm(self, llm: Any, llm_factory: Optional[Callable[[], Any]] = None) -> None:
        """由 LangGraphAgent 注入 LLM 依赖."""
        self._llm = llm
        self._llm_factory = llm_factory

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter("prompt", "string", "完整任务描述，子 Agent 将基于此独立工作"),
            ToolParameter(
                "description",
                "string",
                "3-5 个词的简短说明，用于标识此子任务（如'修复登录bug'、'梳理项目结构'）",
            ),
            ToolParameter(
                "subagent_type",
                "string",
                "子 Agent 类型：coder（默认）、explore（只读）、plan（只规划）",
                required=False,
                default="coder",
                enum=["coder", "explore", "plan"],
            ),
            ToolParameter(
                "resume",
                "string",
                "可选。提供已有子 Agent 实例 ID 以唤回该实例继续任务。与 subagent_type 互斥。",
                required=False,
            ),
            ToolParameter(
                "run_in_background",
                "boolean",
                "是否后台运行，默认 false（前台模式等待完成）",
                required=False,
                default=False,
            ),
        ]

    def execute(
        self,
        prompt: str = "",
        description: str = "",
        subagent_type: str = "coder",
        resume: str = "",
        run_in_background: bool = False,
    ) -> str:
        if not prompt:
            return "[错误] prompt 参数不能为空"
        if not description:
            return "[错误] description 参数不能为空（请提供 3-5 个词的简短说明）"

        if subagent_type not in ("coder", "explore", "plan"):
            return f"[错误] 不支持的 subagent_type: {subagent_type!r}"

        if self._llm is None and self._llm_factory is None:
            return "[错误] Agent 工具未初始化 LLM，无法派发"

        instance_id = resume or None
        return self._manager.dispatch(
            agent_type=subagent_type,
            prompt=prompt,
            llm=self._llm,
            llm_factory=self._llm_factory,
            run_in_background=run_in_background,
            instance_id=instance_id,
        )
