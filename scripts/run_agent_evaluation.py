"""AI Coding Agent 评估脚本.

支持两种模式：
1. 自动模式（默认）：评测员是另一个 LLM，自动与被测 Agent 对话并评判
2. 交互模式（--interactive）：评测员是当前对话中的用户/Kimi Code，每轮暂停等待输入

用法示例:
    # 自动模式
    python scripts/run_agent_evaluation.py \
        --task "请用 Python 实现一个控制台俄罗斯方块游戏" \
        --max-rounds 30 \
        --work-dir ./eval_workspace

    # 交互模式（由当前对话的 Kimi Code 作为评测员）
    python scripts/run_agent_evaluation.py \
        --task "请用 Python 实现一个控制台俄罗斯方块游戏" \
        --max-rounds 30 \
        --work-dir ./eval_workspace \
        --interactive

脚本流程:
1. 初始化被测 Agent（auto_approve=True，自动批准工具调用）
2. 第一轮输入任务描述
3. Agent 处理输入并返回结果
4. 自动模式：评测员 LLM 决定下一步；交互模式：暂停等待人工输入
5. 循环直到任务完成、失败或达到最大轮次
6. 生成评估报告到 evaluation_reports/ 目录
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# 将项目 src 加入路径，支持直接运行脚本
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root / "src"))

from langchain_core.messages import HumanMessage

from ai_coding.agent import AgentService
from ai_coding.config import DEFAULT_LLM_PROVIDER
from ai_coding.llm import create_lc_llm
from ai_coding.logger import setup_logging


class AgentEvaluator:
    """Agent 评估器.

    一端是被测 Agent，另一端可以是自动 LLM 评测员，也可以是当前对话中的用户.
    """

    def __init__(
        self,
        task_description: str,
        max_rounds: int = 20,
        work_dir: str = ".",
        evaluator_model: str = "kimi",
        interactive: bool = False,
    ) -> None:
        self.task_description = task_description
        self.max_rounds = max_rounds
        self.work_dir = str(Path(work_dir).resolve())
        self.evaluator_model = evaluator_model
        self.interactive = interactive

        # 被测 Agent：自动批准工具调用，避免交互式 approval 阻塞
        self.service = AgentService(
            work_dir=self.work_dir,
            llm_provider=DEFAULT_LLM_PROVIDER,
            auto_approve=True,
        )

        # 自动模式下的评测员 LLM
        self.evaluator_llm = create_lc_llm(evaluator_model) if not interactive else None

        self.conversation: List[Dict[str, Any]] = []
        self.current_round = 0
        self.report_dir = project_root / "evaluation_reports"

    def _format_agent_turn(self, agent_output: str) -> str:
        """把 Agent 输出和最近工具调用格式化为评测员可理解的内容."""
        tool_lines: List[str] = []

        history = self.service.get_history()
        recent_messages = history[-10:] if len(history) > 10 else history
        for msg in recent_messages:
            tool_calls = msg.get("tool_calls") or []
            for tc in tool_calls:
                func = tc.get("function", {})
                name = func.get("name", "")
                args = func.get("arguments", {})
                tool_lines.append(f"  - Tool Call: {name}({args})")

        parts = []
        if agent_output.strip():
            parts.append(f"Agent 回复：\n{agent_output.strip()}")
        if tool_lines:
            parts.append("Agent 调用的工具：\n" + "\n".join(tool_lines))

        return "\n\n".join(parts) if parts else "（Agent 没有文本回复，但可能执行了工具）"

    def _build_evaluator_prompt(self, agent_turn_text: str) -> str:
        """构建自动评测员 prompt."""
        history_text = ""
        for turn in self.conversation[-6:]:
            history_text += f"User: {turn['user']}\n"
            history_text += f"Agent: {turn['agent_output'][:500]}\n\n"

        prompt = f"""你是一个严格但友好的测试用户，正在评估一个 AI 编程助手。

你的测试目标：{self.task_description}

评估规则（必须遵守）：
1. 你会看到助手最近一轮的回复和工具调用情况。
2. 如果助手问你问题或请求确认，请像真实用户一样简短回答。
3. 如果助手已经完整、正确地完成任务，请只回复一行：
   [TASK_COMPLETE] 任务已完成：简要说明
4. 如果助手明显失败、拒绝、长时间无进展或无法修复，请只回复一行：
   [TASK_FAILED] 失败原因
5. 否则，请给出下一步指令或反馈，推动任务完成。指令要具体、可操作。
6. 每次只输出一条用户消息，不要输出额外解释、分析、markdown 或代码块。

当前轮次：{self.current_round}/{self.max_rounds}

最近对话历史：
{history_text}

助手最新一轮：
{agent_turn_text}

请输出你的下一条用户消息：
"""
        return prompt

    def _get_auto_evaluator_response(self, prompt: str) -> str:
        """调用自动评测员 LLM 获取回复."""
        response = self.evaluator_llm.invoke([HumanMessage(content=prompt)])
        return response.content if hasattr(response, "content") else str(response)

    def _get_interactive_input(self, agent_turn_text: str) -> str:
        """交互模式：等待人工评测员输入下一步."""
        print("\n" + "-" * 60)
        print("[交互模式] 请作为评测员输入下一步用户消息")
        print("提示：")
        print("  - 直接输入消息，Agent 会在下一轮收到")
        print("  - 输入 [DONE] 表示任务完成")
        print("  - 输入 [FAIL] 表示任务失败")
        print("  - 输入 [QUIT] 退出评估")
        print("-" * 60)

        while True:
            try:
                user_input = input(">>> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n用户取消输入，退出评估。")
                return "[QUIT]"

            if not user_input:
                print("输入不能为空，请重新输入。")
                continue
            return user_input

    def _save_report(self, status: str, reason: str) -> Path:
        """保存评估报告."""
        self.report_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        safe_task = "".join(c if c.isalnum() or c in "_-" else "_" for c in self.task_description[:30])
        report_path = self.report_dir / f"eval_{status}_{safe_task}_{timestamp}.json"

        report = {
            "task": self.task_description,
            "status": status,
            "reason": reason,
            "max_rounds": self.max_rounds,
            "actual_rounds": self.current_round,
            "work_dir": self.work_dir,
            "evaluator_model": "human" if self.interactive else self.evaluator_model,
            "conversation": self.conversation,
        }

        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        return report_path

    def run(self) -> Tuple[bool, str]:
        """运行评估循环."""
        current_user_input = self.task_description

        print(f"开始评估任务：{self.task_description}")
        print(f"工作目录：{self.work_dir}")
        print(f"最大轮次：{self.max_rounds}")
        print(f"评测模式：{'交互模式（人工评测员）' if self.interactive else '自动模式（' + self.evaluator_model + '）'}")
        print("=" * 60)

        for i in range(self.max_rounds):
            self.current_round = i + 1

            print(f"\n--- Round {self.current_round}/{self.max_rounds} ---")
            print(f"User: {current_user_input[:300]}")

            # Agent 处理用户输入
            agent_output = self.service.send_message(current_user_input)
            agent_turn_text = self._format_agent_turn(agent_output)

            print(f"Agent: {agent_output[:500]}")
            if "Agent 调用的工具：" in agent_turn_text:
                tools_part = agent_turn_text.split("Agent 调用的工具：", 1)[-1].strip()
                print(f"Tools: {tools_part[:300]}")

            self.conversation.append({
                "round": self.current_round,
                "user": current_user_input,
                "agent_output": agent_output,
                "agent_turn_text": agent_turn_text,
            })

            # 获取评测员下一步输入
            if self.interactive:
                evaluator_response = self._get_interactive_input(agent_turn_text)
            else:
                evaluator_prompt = self._build_evaluator_prompt(agent_turn_text)
                evaluator_response = self._get_auto_evaluator_response(evaluator_prompt).strip()
                print(f"Evaluator: {evaluator_response[:300]}")

            # 处理特殊指令
            upper_response = evaluator_response.upper()
            if upper_response.startswith("[DONE]") or "[TASK_COMPLETE]" in upper_response:
                reason = evaluator_response.split("]", 1)[-1].strip() if "]" in evaluator_response else "任务完成"
                report_path = self._save_report("completed", reason)
                print(f"\n[完成] {reason}")
                print(f"报告已保存：{report_path}")
                return True, reason

            if upper_response.startswith("[FAIL]") or "[TASK_FAILED]" in upper_response:
                reason = evaluator_response.split("]", 1)[-1].strip() if "]" in evaluator_response else "任务失败"
                report_path = self._save_report("failed", reason)
                print(f"\n[失败] {reason}")
                print(f"报告已保存：{report_path}")
                return False, reason

            if upper_response.startswith("[QUIT]"):
                report_path = self._save_report("manual_quit", "用户手动退出")
                print(f"\n[退出] 用户手动退出")
                print(f"报告已保存：{report_path}")
                return False, "用户手动退出"

            current_user_input = evaluator_response

        # 达到最大轮次
        report_path = self._save_report("max_rounds_reached", "达到最大轮次，任务未完成")
        print(f"\n[截止] 达到最大轮次 {self.max_rounds}，任务未完成")
        print(f"报告已保存：{report_path}")
        return False, "达到最大轮次，任务未完成"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI Coding Agent 评估脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例：
  # 自动模式
  python scripts/run_agent_evaluation.py --task "实现一个控制台俄罗斯方块"

  # 交互模式（由当前对话的 Kimi Code 作为评测员）
  python scripts/run_agent_evaluation.py --task "实现一个控制台俄罗斯方块" --interactive
""",
    )
    parser.add_argument(
        "--task",
        required=True,
        help="任务描述，例如：让 Agent 做一个俄罗斯方块游戏",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=20,
        help="最大对话轮次（默认 20）",
    )
    parser.add_argument(
        "--work-dir",
        default=".",
        help="Agent 工作目录（默认当前目录）",
    )
    parser.add_argument(
        "--evaluator-model",
        default="kimi",
        help="自动模式下评测员使用的模型：kimi | openai | mock（默认 kimi）",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="交互模式：每轮暂停，等待人工评测员输入下一步",
    )
    args = parser.parse_args()

    setup_logging()

    # 确保工作目录存在
    work_dir_path = Path(args.work_dir).resolve()
    work_dir_path.mkdir(parents=True, exist_ok=True)

    evaluator = AgentEvaluator(
        task_description=args.task,
        max_rounds=args.max_rounds,
        work_dir=str(work_dir_path),
        evaluator_model=args.evaluator_model,
        interactive=args.interactive,
    )

    success, result = evaluator.run()
    print("\n" + "=" * 60)
    print(f"最终结果：{'成功' if success else '失败'}")
    print(f"说明：{result}")


if __name__ == "__main__":
    main()
