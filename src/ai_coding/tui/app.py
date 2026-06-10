"""AI Coding TUI 应用 - Claude Code 风格.

采用 Textual 构建全屏终端界面，支持：
- 消息气泡（用户/AI/工具/错误区分样式）
- Markdown 代码块语法高亮
- 底部固定输入栏
- 审批弹窗
- 流式输出（消息传递机制）
"""

import os
from pathlib import Path
from typing import Optional

from textual.app import App, ComposeResult
from textual.message import Message

from ai_coding.config import DEFAULT_LLM_PROVIDER
from ai_coding.llm import create_lc_llm
from ai_coding.agent import SessionManager
from ai_coding.logger import setup_logging
from ai_coding.tools import DEFAULT_TOOLS
from ai_coding.tui.widgets import (
    ApprovalModal,
    ChatLog,
    InputBar,
    MessageType,
    StatusBar,
)


class AgentResponse(Message):
    """Agent 响应消息（跨线程传递）."""

    def __init__(self, content: str, is_error: bool = False) -> None:
        self.content = content
        self.is_error = is_error
        super().__init__()


class AICodingApp(App):
    """AI Coding TUI 主应用."""

    CSS_PATH = Path(__file__).parent / "styles.css"

    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+q", "quit", "Quit"),
    ]

    def __init__(
        self,
        work_dir: str,
        auto_approve: bool = False,
        session_id: Optional[str] = None,
    ) -> None:
        self.work_dir = work_dir
        self.auto_approve = auto_approve
        self.target_session_id = session_id
        self.sm: Optional[SessionManager] = None
        super().__init__()

    def compose(self) -> ComposeResult:
        """组装 UI."""
        yield StatusBar(id="status-bar")
        yield ChatLog(id="chat-log")
        yield InputBar(id="input-bar")

    def on_mount(self) -> None:
        """应用挂载时初始化."""
        setup_logging()
        os.chdir(self.work_dir)

        self.sm = SessionManager(
            llm_factory=lambda: create_lc_llm(DEFAULT_LLM_PROVIDER),
            tools=DEFAULT_TOOLS,
            auto_approve=self.auto_approve,
            work_dir=self.work_dir,
        )

        if self.target_session_id:
            self.sm.switch(self.target_session_id)

        status_bar = self.query_one("#status-bar", StatusBar)
        status_bar.set_model(DEFAULT_LLM_PROVIDER)

        self._welcome_message()

    def _welcome_message(self) -> None:
        """显示欢迎信息."""
        chat_log = self.query_one("#chat-log", ChatLog)
        chat_log.add_message(
            f"Project: {self.work_dir}\n"
            f"Provider: {DEFAULT_LLM_PROVIDER}\n\n"
            "Commands:\n"
            "  /help     - Show help\n"
            "  /new      - Create new session\n"
            "  /session  - Session management\n"
            "  /exit     - Exit",
            MessageType.SYSTEM,
        )

    def on_input_bar_submitted(self, event: InputBar.Submitted) -> None:
        """处理输入提交."""
        command = event.value.strip()
        if not command:
            return

        chat_log = self.query_one("#chat-log", ChatLog)
        chat_log.add_message(command, MessageType.USER)

        if command in ("/exit", "/quit"):
            self.exit()
            return

        if command == "/help":
            chat_log.add_message(
                "Available commands:\n"
                "  /help              - Show this help\n"
                "  /new               - Create a new session\n"
                "  /session list      - List all sessions\n"
                "  /session switch ID - Switch session\n"
                "  /status            - Show context usage\n"
                "  /exit              - Exit\n"
                "  (Any other text is sent to the Agent)",
                MessageType.SYSTEM,
            )
            return

        if command == "/new":
            sid = self.sm.create()
            chat_log.add_message(f"Created session: {sid}", MessageType.SYSTEM)
            return

        if command.startswith("/session "):
            self._handle_session_command(command, chat_log)
            return

        # 发送给 Agent（在工作线程中执行）
        status_bar = self.query_one("#status-bar", StatusBar)
        status_bar.state_text = "Thinking..."
        self.run_worker(self._run_agent, command, thread=True)

    def _handle_session_command(self, command: str, chat_log: ChatLog) -> None:
        """处理 /session 子命令."""
        parts = command.split()
        if len(parts) < 2:
            chat_log.add_message("Usage: /session list | switch ID", MessageType.SYSTEM)
            return

        sub = parts[1].lower()
        if sub == "list":
            sessions = self.sm.list()
            if not sessions:
                chat_log.add_message("No sessions.", MessageType.SYSTEM)
                return
            lines = ["Sessions:"]
            for s in sessions:
                marker = "*" if s["is_current"] else " "
                lines.append(
                    f"  [{marker}] {s['session_id']}  {s['name']}  "
                    f"({s['message_count']} msgs)"
                )
            chat_log.add_message("\n".join(lines), MessageType.SYSTEM)
        elif sub == "switch":
            if len(parts) < 3:
                chat_log.add_message("Usage: /session switch <ID>", MessageType.SYSTEM)
                return
            sid = parts[2]
            if self.sm.switch(sid):
                chat_log.add_message(f"Switched to: {sid}", MessageType.SYSTEM)
            else:
                chat_log.add_message(f"Session not found: {sid}", MessageType.ERROR)
        else:
            chat_log.add_message(f"Unknown subcommand: {sub}", MessageType.ERROR)

    def _run_agent(self, prompt: str) -> None:
        """在工作线程中运行 Agent."""
        agent = self.sm.get_current_agent()
        if agent is None:
            self.post_message(AgentResponse("No active session", is_error=True))
            return

        try:
            result = agent.run(prompt)
            self.post_message(AgentResponse(result))
        except Exception as e:
            self.post_message(AgentResponse(str(e), is_error=True))

    def on_agent_response(self, message: AgentResponse) -> None:
        """处理 Agent 响应."""
        chat_log = self.query_one("#chat-log", ChatLog)
        msg_type = MessageType.ERROR if message.is_error else MessageType.AI
        chat_log.add_message(message.content, msg_type)

        status_bar = self.query_one("#status-bar", StatusBar)
        status_bar.state_text = "Ready"

    async def show_approval(self, tool_name: str, args: dict) -> bool:
        """显示审批弹窗并等待结果."""
        return await self.push_screen_wait(ApprovalModal(tool_name, args))


def run_tui(
    work_dir: str,
    auto_approve: bool = False,
    session_id: Optional[str] = None,
) -> None:
    """启动 TUI."""
    app = AICodingApp(
        work_dir=work_dir,
        auto_approve=auto_approve,
        session_id=session_id,
    )
    app.run()
