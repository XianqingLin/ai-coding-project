"""AI Coding TUI 应用.

使用 Textual 构建全屏终端界面，支持会话管理、消息历史和 Agent 交互.
"""

from pathlib import Path
from typing import Optional

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Input, Label, ListItem, ListView, RichLog

from ai_coding.config import DEFAULT_LLM_PROVIDER
from ai_coding.llm import create_lc_llm
from ai_coding.agent import SessionManager
from ai_coding.logger import setup_logging
from ai_coding.tools import DEFAULT_TOOLS


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
        with Horizontal(id="main"):
            # 左侧会话列表
            with Vertical(id="sidebar"):
                yield Label("Sessions", id="sidebar-title")
                yield ListView(id="session-list")

            # 右侧主区域
            with Vertical(id="content"):
                # 顶部状态栏
                with Horizontal(id="status-bar"):
                    yield Label("AI Coding", id="app-title")
                    yield Label(DEFAULT_LLM_PROVIDER, id="model-label")
                    yield Label("Ready", id="status-label")

                # 消息日志区
                yield RichLog(id="message-log", wrap=True, highlight=True)

                # 底部输入区
                yield Input(placeholder=">>> ", id="command-input")

    def on_mount(self) -> None:
        """应用挂载时初始化."""
        setup_logging()

        self.sm = SessionManager(
            llm_factory=lambda: create_lc_llm(DEFAULT_LLM_PROVIDER),
            tools=DEFAULT_TOOLS,
            auto_approve=self.auto_approve,
            work_dir=self.work_dir,
        )

        if self.target_session_id:
            self.sm.switch(self.target_session_id)

        self._refresh_session_list()
        self._welcome_message()

    def _refresh_session_list(self) -> None:
        """刷新会话列表."""
        list_view = self.query_one("#session-list", ListView)
        list_view.clear()
        sessions = self.sm.list()
        for s in sessions:
            marker = "*" if s["is_current"] else " "
            label = f"{marker} {s['name']}"
            item = ListItem(Label(label))
            # type: ignore[attr-defined]
            item.session_id = s["session_id"]
            list_view.append(item)

    def _welcome_message(self) -> None:
        """显示欢迎信息."""
        log = self.query_one("#message-log", RichLog)
        log.write(f"Project: {self.work_dir}")
        log.write(f"Provider: {DEFAULT_LLM_PROVIDER}")
        log.write("")
        log.write("Commands:")
        log.write("  /help     - Show help")
        log.write("  /new      - Create new session")
        log.write("  /session  - Session management")
        log.write("  /exit     - Exit")
        log.write("─" * 40)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """处理输入提交."""
        input_widget = self.query_one("#command-input", Input)
        command = event.value.strip()
        input_widget.value = ""

        if not command:
            return

        log = self.query_one("#message-log", RichLog)
        log.write(f"[b]>>>[/b] {command}")

        if command in ("/exit", "/quit"):
            self.exit()
            return

        if command == "/help":
            log.write("[b]Available commands:[/b]")
            log.write("  /help              - Show this help")
            log.write("  /new               - Create a new session")
            log.write("  /session list      - List all sessions")
            log.write("  /session switch ID - Switch session")
            log.write("  /status            - Show context usage")
            log.write("  /verbose           - Toggle verbose mode")
            log.write("  /exit              - Exit")
            log.write("  (Any other text is sent to the Agent)")
            return

        if command == "/new":
            sid = self.sm.create()
            self._refresh_session_list()
            log.write(f"[green]Created session: {sid}[/green]")
            return

        if command.startswith("/session "):
            self._handle_session_command(command, log)
            return

        # 发送给 Agent（在工作线程中执行）
        self.query_one("#status-label", Label).update("Thinking...")
        self.run_worker(self._run_agent, command, thread=True)

    def _handle_session_command(self, command: str, log: RichLog) -> None:
        """处理 /session 子命令."""
        parts = command.split()
        if len(parts) < 2:
            log.write("[red]Usage: /session list | switch ID[/red]")
            return

        sub = parts[1].lower()
        if sub == "list":
            sessions = self.sm.list()
            if not sessions:
                log.write("No sessions.")
                return
            for s in sessions:
                marker = "*" if s["is_current"] else " "
                log.write(
                    f"  [{marker}] {s['session_id']}  {s['name']}  "
                    f"({s['message_count']} msgs)"
                )
        elif sub == "switch":
            if len(parts) < 3:
                log.write("[red]Usage: /session switch <ID>[/red]")
                return
            sid = parts[2]
            if self.sm.switch(sid):
                self._refresh_session_list()
                log.write(f"[green]Switched to: {sid}[/green]")
            else:
                log.write(f"[red]Session not found: {sid}[/red]")
        else:
            log.write(f"[red]Unknown /session subcommand: {sub}[/red]")

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
        log = self.query_one("#message-log", RichLog)
        if message.is_error:
            log.write(f"[red][Error] {message.content}[/red]")
        else:
            log.write(f"[cyan][Assistant] {message.content}[/cyan]")
        self.query_one("#status-label", Label).update("Ready")

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """切换会话."""
        sid = getattr(event.item, "session_id", None)
        if sid and self.sm:
            if self.sm.switch(sid):
                self._refresh_session_list()
                log = self.query_one("#message-log", RichLog)
                log.write(f"[green]Switched to session: {sid}[/green]")


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
