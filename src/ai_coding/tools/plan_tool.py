"""Plan 工具.

Agent 调用此工具提交修改计划，标志着从探索阶段进入修改阶段.
"""

from typing import List

from ai_coding.tools.base import Tool, ToolParameter


class PlanTool(Tool):
    """提交修改计划，从探索阶段进入修改阶段."""

    name = "plan"
    description = (
        "提交修改计划，正式进入修改阶段。\n"
        "在读了几(3-5)个关键文件、理解了问题所在后，你就应该调用此工具。\n"
        "计划不需要完美或详尽——写清楚要改哪些文件、大致怎么改即可。\n"
        "你可以在后续的 str_replace_file 执行中随时修正和调整计划。\n"
        "调用 plan 后请立即开始 str_replace_file 修改，不要继续探索。"
    )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                "plan",
                "string",
                "详细的修改计划。包括：1) 要修改的文件列表；2) 每个文件的具体修改内容；3) 修改的理由。",
            ),
        ]

    def execute(self, plan: str) -> str:
        # 实际的状态切换由 LangGraphAgent.tools_node 处理
        # 这里返回确认信息
        return (
            f"[计划已记录] 你已正式进入修改阶段。\n"
            f"你的计划：\n{'='*50}\n{plan}\n{'='*50}\n\n"
            f"【立即执行】你的下一步**必须**是 str_replace_file 或 write_file，"
            f"不允许再使用 read_file/grep/list_dir。\n"
            f"请现在就选择计划中的一个文件，使用 str_replace_file 开始修改。"
        )
