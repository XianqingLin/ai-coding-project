# AI Coding

[![CI](https://github.com/XianqingLin/ai-coding-project/actions/workflows/ci.yml/badge.svg)](https://github.com/XianqingLin/ai-coding-project/actions)
[![Coverage](https://codecov.io/gh/XianqingLin/ai-coding-project/branch/master/graph/badge.svg)](https://codecov.io/gh/XianqingLin/ai-coding-project)

一个基于 AI 的**控制台交互式编程助手**，参考 [Claude Code](https://github.com/anthropropics/claude-code) 设计。

通过自然语言与 AI 对话，让它帮你读取文件、编辑代码、执行命令、回答问题。

## 功能特性

- 🤖 **ReAct Agent 架构** — AI 能思考、决策、调用工具完成任务
- ⚡ **流式输出** — AI 回复逐字实时显示，无需等待
- 🛠️ **工具系统** — 读取/写入/编辑文件、列出目录、执行命令、代码搜索
- 🎨 **语法高亮** — 代码块自动高亮显示
- 📝 **日志记录** — 自动记录操作日志到 `logs/` 目录
- 🔌 **多模型支持** — 支持 Kimi (K2.6)、OpenAI、Mock 模式

## 快速开始

### 环境要求

- Python >= 3.10

### 安装

```bash
# 克隆仓库
git clone https://github.com/yourusername/ai-coding.git
cd ai-coding

# 创建虚拟环境
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

# 安装依赖（推荐 editable 模式）
pip install -e .
```

### 配置

复制 `.env.example` 为 `.env`，填入你的 API Key：

```bash
cp .env.example .env
```

编辑 `.env`：

```env
# Kimi (月之暗面) — 推荐
KIMI_API_KEY=your_kimi_api_key_here
KIMI_BASE_URL=https://api.moonshot.cn/v1
KIMI_MODEL=kimi-k2.6

# 默认使用的 LLM 提供商: kimi | openai | mock
DEFAULT_LLM_PROVIDER=kimi
```

> 获取 API Key：https://platform.moonshot.cn/

### 运行

```bash
# Windows
$env:PYTHONIOENCODING="utf-8"
$env:PYTHONPATH="src"
python -m ai_coding

# 或简写
PYTHONIOENCODING=utf-8 PYTHONPATH=src python -m ai_coding
```

启动后你会看到：

```
[信息] 使用 LLM 提供商: kimi
[信息] 模型: kimi-k2.6
+------------------------------------+
|     AI Coding Assistant            |
+------------------------------------+
|  输入内容开始对话                  |
|  /help 查看命令  /exit 退出        |
+------------------------------------+
>>> 帮我查看 README.md
[Agent 思考中...]
README.md 的内容如下：
# AI Coding
...
```

## 使用示例

### 查看文件

```
>>> 帮我查看 src/main.py 的内容
```

### 列出目录

```
>>> 列出当前目录结构
```

### 执行命令

```
>>> 运行 pytest 测试
```

### 编辑文件

```
>>> 把 README.md 中的 "Python >= 3.10" 改成 "Python >= 3.11"
```

### DeepSWE 单任务评估

```bash
# 运行单个 DeepSWE 任务
python scripts/run_single.py --task abs-module-cache-flags

# 单步调试模式
python scripts/run_single.py --task abs-module-cache-flags --step

# 使用自定义提示
python scripts/run_single.py --task abs-module-cache-flags --prompt prompts/v2.txt
```

### 内置命令

| 命令 | 功能 |
|------|------|
| `/help` | 显示帮助信息 |
| `/prompt` | 显示当前系统提示 |
| `/history` | 查看对话历史 |
| `/session list` | 列出所有会话 |
| `/session switch ID` | 切换会话 |
| `/session rm ID` | 删除会话 |
| `/session rename ID NAME` | 重命名会话 |
| `/new` | 创建新会话 |
| `/clear` | 清屏 |
| `/exit` | 退出程序 |

## 项目结构

```
ai-coding/
├── logs/                   # 自动生成的日志文件
├── data/                   # 会话持久化数据
├── scripts/                # 评估与辅助脚本
├── tests/                  # 单元测试
├── src/
│   └── ai_coding/
│       ├── __init__.py     # 包入口
│       ├── __main__.py     # python -m ai_coding
│       ├── cli.py          # 命令行入口
│       ├── config.py       # 环境变量配置
│       ├── logger.py       # 日志系统
│       ├── mock_llm.py     # Mock LLM（离线测试）
│       ├── agent/          # Agent 核心
│       │   ├── core.py         # LangGraph ReAct Agent 运行时
│       │   ├── service.py      # Agent 统一服务接口
│       │   ├── events.py       # 标准事件类型
│       │   ├── session.py      # 多会话管理
│       │   ├── state.py        # AgentState 定义
│       │   ├── context_compressor.py  # 上下文压缩
│       │   ├── sub_agent_manager.py   # 子 Agent 管理
│       │   └── nodes/          # 图节点（llm、tools、approval、edges）
│       ├── llm/            # LLM 封装
│       │   ├── kimi_chat.py    # KimiChatOpenAI（支持 reasoning_content）
│       │   └── lc_llm.py       # LLM 工厂（Kimi/OpenAI/Mock）
│       ├── persistence/    # 持久化存储
│       │   ├── storage.py      # 存储引擎
│       │   ├── serializer.py   # 状态序列化
│       │   └── config.py       # 存储配置
│       ├── prompts/        # 系统提示模板
│       └── tools/          # 工具系统
│           ├── base.py         # Tool / ToolRegistry
│           ├── file_tools.py   # 文件操作（read/write/edit/grep/glob/list_dir）
│           ├── shell_tools.py  # Shell 命令执行（含后台任务）
│           ├── task_tools.py   # 后台任务管理（task_list/output/stop）
│           ├── plan_tools.py   # Plan 模式工具
│           ├── collaboration_tools.py  # 协作工具
│           └── todo_tool.py    # 任务列表
├── .env                    # 环境变量（API Key 等）
├── .env.example            # 环境变量模板
├── .gitignore
├── pyproject.toml          # 项目配置
└── README.md               # 本文件
```

## 开发

### 运行测试

```bash
pytest
```

### 使用 Mock 模式（不消耗 API）

```bash
# 修改 .env
DEFAULT_LLM_PROVIDER=mock
```

### 日志位置

```
logs/ai-coding-YYYY-MM-DD.log
```

## 技术栈

| 组件 | 说明 |
|------|------|
| ReAct Agent | 推理-行动循环架构 |
| Function Calling | OpenAI 兼容的工具调用协议 |
| `openai` | LLM API 客户端 |
| `rich` | 控制台美化、Markdown/代码高亮 |
| `python-dotenv` | 环境变量管理 |
| `pytest` | 单元测试 |

## 许可证

MIT License
