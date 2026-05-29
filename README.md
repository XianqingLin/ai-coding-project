# AI Coding

一个基于 AI 的**控制台交互式编程助手**，参考 [Claude Code](https://github.com/anthropropics/claude-code) 设计。

通过自然语言与 AI 对话，让它帮你读取文件、编辑代码、执行命令、回答问题。

## 功能特性

- 🤖 **ReAct Agent 架构** — AI 能思考、决策、调用工具完成任务
- ⚡ **流式输出** — AI 回复逐字实时显示，无需等待
- 🛠️ **工具系统** — 读取/写入/编辑文件、列出目录、执行命令
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

# 安装依赖
pip install -r requirements.txt
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

### 内置命令

| 命令 | 功能 |
|------|------|
| `/help` | 显示帮助信息 |
| `/tools` | 列出可用工具 |
| `/history` | 查看对话历史 |
| `/new` | 开启新会话（清空历史） |
| `/verbose` | 切换详细模式 |
| `/stream` | 切换流式输出模式 |
| `/clear` | 清屏 |
| `/exit` | 退出程序 |

## 项目结构

```
ai-coding/
├── logs/                   # 自动生成的日志文件
├── src/
│   └── ai_coding/
│       ├── __init__.py     # 包入口
│       ├── __main__.py     # python -m ai_coding
│       ├── main.py         # 程序入口
│       ├── agent.py        # ReAct Agent 核心
│       ├── llm.py          # LLM 接口（Kimi/OpenAI/Mock）
│       ├── repl.py         # 控制台交互（流式/高亮）
│       ├── config.py       # 环境变量配置
│       ├── logger.py       # 日志系统
│       └── tools/          # 工具系统
│           ├── base.py     # Tool / ToolRegistry
│           └── file_tools.py  # 文件操作工具
├── tests/                  # 测试文件
├── docs/
│   └── ROADMAP.md          # 开发路线图
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
