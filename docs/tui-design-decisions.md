# TUI 设计决策文档

> 创建于 2026-06-10
> 用于 `/compact` 后恢复上下文

---

## 1. 核心决策

### 1.1 技术栈选择

| 方案 | 状态 | 原因 |
|------|------|------|
| **Textual** | ❌ 已废弃 | 全屏 TUI，会清屏接管终端，不符合"在原终端继续操作"的需求 |
| **Rich + prompt_toolkit** | ✅ 选定 | 在终端中原地渲染，不清屏，符合 Kimi Code CLI / Claude Code CLI 风格 |

### 1.2 仿照对象

| 来源 | 采用内容 | 不采用 |
|------|---------|--------|
| **Kimi Code CLI** | 骨架布局（顶部状态栏 + 中间历史 + 底部输入框）、终端原生风格 | 朴素文本输出 |
| **Claude Code CLI** | 代码语法高亮、Diff 渲染（红绿着色） | 气泡样式（在终端中实现成本过高） |

### 1.3 关键区别

- **Textual**：`App.run()` 会清屏，用户退出后看不到之前的输出
- **Rich + prompt_toolkit**：在终端中动态输出，保留所有历史，像 `htop` vs `ls` 的区别

---

## 2. 目标界面设计

```
PS C:\...\ai-coding> python -m ai_coding.cli chat

Project: C:\...\ai-coding   Provider: kimi   Ready
───────────────────────────────────────────────────

>>> 帮我写个排序算法
[Assistant] 好的，我来实现一个快速排序：

┌────────────────────────────────────────────┐
│ def quicksort(arr):                        │
│     if len(arr) <= 1:                      │
│         return arr                         │
│     pivot = arr[len(arr) // 2]             │
│     ...                                    │
└────────────────────────────────────────────┘

[Tool] write_file(quicksort.py)
[OK] File created

>>> _  ← prompt_toolkit 输入框（Shift+Enter 换行）
```

---

## 3. 当前代码状态

### 3.1 已废弃的代码（待删除）

- `src/ai_coding/tui/app.py` — Textual App（AICodingApp）
- `src/ai_coding/tui/widgets.py` — Textual 组件（MessageBubble, ChatLog 等）
- `src/ai_coding/tui/styles.css` — Textual CSS

### 3.2 需要新建的文件

| 文件 | 说明 |
|------|------|
| `src/ai_coding/tui/render.py` | Rich 渲染函数（消息、代码块、Diff） |
| `src/ai_coding/tui/shell.py` | prompt_toolkit 输入框和事件循环 |
| `src/ai_coding/tui/app.py` | 主应用（替换旧的 Textual App） |

### 3.3 保留的文件

- `src/ai_coding/tui/__init__.py` — 导出入口
- `src/ai_coding/cli.py` — CLI 命令路由（ask/chat/session）
- `src/ai_coding/main.py` — REPL 逻辑（chat --no-tui 用）

---

## 4. 组件设计

### 4.1 Rich 渲染层

```python
# render.py

def render_message(content: str, role: str) -> RenderableType:
    """渲染单条消息."""
    # 用户消息：蓝色前缀
    # AI 消息：绿色前缀 + 解析代码块
    # 工具调用：黄色前缀
    # 错误：红色前缀

def render_code_block(code: str, lang: str) -> Panel:
    """代码块语法高亮."""
    # 使用 rich.Syntax

def render_diff(diff_text: str) -> Table:
    """Diff 红绿着色."""
    # + 行绿色，- 行红色
```

### 4.2 prompt_toolkit 输入层

```python
# shell.py

from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings

class TUIRunner:
    def __init__(self):
        self.session = PromptSession(
            message=">>> ",
            multiline=True,  # Shift+Enter 换行
            key_bindings=self._create_key_bindings(),
        )
    
    def run(self, sm: SessionManager):
        """主循环."""
        while True:
            text = self.session.prompt()
            result = sm.get_current_agent().run(text)
            # 用 Rich 渲染结果
```

---

## 5. 依赖变更

### 5.1 待安装

```
pip install prompt-toolkit
```

### 5.2 待移除（pyproject.toml）

```toml
# 删除 textual 依赖
textual>=0.50
```

### 5.3 保留

```toml
rich>=13.0    # 已存在，用于代码高亮
```

---

## 6. 工作量估算

| 任务 | 预估时间 |
|------|---------|
| 删除旧 Textual 代码 | 10 分钟 |
| 创建 Rich render.py | 1 小时 |
| 创建 prompt_toolkit shell.py | 1.5 小时 |
| 整合到 cli.py chat 命令 | 30 分钟 |
| 更新 pyproject.toml | 10 分钟 |
| **总计** | **~3 小时** |

---

## 7. 重要 Git 提交

| 提交 | 说明 |
|------|------|
| `f245228` | feat: add Textual TUI for chat command（已废弃，待回滚）|
| `5d47cee` | feat: add CLI with typer（保留） |
| `8496a54` | chore: remove all test scripts（保留） |

---

## 8. 待办事项（/compact 后接续）

- [ ] 删除旧 Textual 代码（tui/app.py, tui/widgets.py, tui/styles.css）
- [ ] 创建 `src/ai_coding/tui/render.py`（Rich 渲染）
- [ ] 创建 `src/ai_coding/tui/shell.py`（prompt_toolkit 输入循环）
- [ ] 重写 `src/ai_coding/tui/app.py`（主应用入口）
- [ ] 更新 `pyproject.toml`：移除 textual，添加 prompt-toolkit
- [ ] 本地测试 TUI 启动
