# 🛡️ 净源哨兵 —— AI 训练数据投毒智能检测与防御

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Tests: 78 passed](https://img.shields.io/badge/Tests-78%20passed-brightgreen.svg)](tool_server/tests/)

让大模型做语义裁决，9 个专用工具做大模型看不见的技术检测，**Agent 自主编排完整闭环**（检测 → 判定 → 处置 → 验证）。自建 500 条 benchmark 上 **F1 = 100.00%、零误报**。

> 本项目为挑战杯（XH-202614「AI+安全大模型平台的智能体研究」）参赛作品。核心思路：纯 LLM 有语义无「眼睛」（漏检标签投毒、无全局分布视野），传统工具有「眼睛」无「脑子」（13 个模板误报），**Agent 融合二者 = 100/100/0**。

## 架构

```
┌─────────────────────────────────────────────────────────┐
│                    前端 frontend.html                    │
│          （上传数据 → 查看报告 / 推理链 / 知识库）          │
└────────────────────────┬────────────────────────────────┘
                         │ HTTP
┌────────────────────────▼────────────────────────────────┐
│                  Dify Agent（LLM 中枢）                   │
│   system_prompt_v6.md · DeepSeek · ReAct 多步推理        │
│   自主决定：调用哪些工具、何时隔离、何时人工复核           │
└────────────────────────┬────────────────────────────────┘
                         │ MCP (OpenAPI 工具协议)
┌────────────────────────▼────────────────────────────────┐
│            FastAPI 工具服务器（127.0.0.1:5050）           │
│  9 个自研工具：                                          │
│  🔍 编码层  unicode-scan（零宽字符/同形字）               │
│  📊 分布层  embedding-scan（bge-m3 聚类离群）             │
│  🏷️ 标签层  label-check（标签-内容矛盾）                  │
│  📚 知识层  kb-search（RAG 检索 105 条攻击模式）           │
│  ⚙️ 处置层  clean-data / quarantine / verify-clean        │
└───────────────┬─────────────────────────┬────────────────┘
                │                         │
     ┌──────────▼──────────┐   ┌──────────▼──────────┐
     │ knowledge_base/ 8文件 │   │  Ollama bge-m3 嵌入 │
     └─────────────────────┘   └─────────────────────┘
```

架构图：[docs/arch_design.png](docs/arch_design.png) · [docs/arch_flow.png](docs/arch_flow.png)

## 为什么有效：LLM 盲区 × 工具职责一一对应

| 攻击类型 | 纯 LLM | 工具补齐 |
|---|---|---|
| 标签翻转投毒（好评文本配差评标签） | 0/8 全盲 | label-check 核对标签-情感矛盾 |
| 语义离群投毒（处方/法条/菜谱混入训练集） | 0/8 全盲 | embedding-scan 全局聚类 |
| 同形字伪装指令（西里尔字母混淆） | 8/8 | unicode-scan 确定性编码检测 |
| 零宽字符拆散敏感词 | 8/8 | unicode-scan 确定性编码检测 |

Agent 的价值在于**编排**：自主决定对哪些样本调哪些工具，融合语义理解与确定性信号，输出可审计的逐条判定 + 推理链。

## 系统要求

- Windows 10/11（或任意支持 Docker 的环境）
- Python 3.10+
- [Ollama](https://ollama.com) + `bge-m3` 模型（嵌入）
- Docker（运行 Dify）+ DeepSeek API Key（或任意支持工具调用的 LLM）

## 快速开始（6 步）

### 1. 启动 Ollama

```bash
ollama serve        # 或直接启动 Ollama 桌面应用
ollama pull bge-m3  # 首次使用需拉取嵌入模型
```

### 2. 启动工具服务器

```bash
cd tool_server
pip install -r requirements.txt
python server.py        # 监听 http://127.0.0.1:5050
```

验证：浏览器打开 `http://127.0.0.1:5050/docs` 应看到 9 个 `/tools/*` 端点；`http://127.0.0.1:5050/health` 返回 `净源哨兵 Tool Server v6.2`。

### 3. 配置 Dify Agent

- Docker 启动 Dify，创建 **Agent** 类型应用（不是 Chatflow）
- **系统提示词**：粘贴 `agent/system_prompt_v6.md` 全文（⚠️ 修改提示词文件后必须同步到这里才生效）
- **工具注册**：用 `tool_server/openapi_dify.yaml` 导入 9 个工具
- **模型**：DeepSeek-V4-Pro（或任意支持工具调用的模型）；**最大 token 建议 ≥ 8192**（50 条/批的完整输出实测约 4900 字符，设小了会被截断）
- 记录应用的 API Key（前端与评测共用）

### 4. 知识库

**无需 Dify 侧索引**——Agent 的知识检索由工具服务器自带的 `kb-search` 工具完成，它直接读取项目内的 `knowledge_base/` 目录（8 个文件、105 条结构化攻击模式），向量部分通过 Ollama 的 bge-m3 计算。

两个前提：
- 保持目录结构不变（`kb-search` 按相对路径 `../../../knowledge_base` 定位文件）
- Ollama 中已加载 bge-m3（第 1 步已做）

### 5. 打开前端

浏览器打开 `frontend.html`，配置 API Key（二选一）：

- **改代码**：编辑 `frontend.html` 顶部 `KEY='YOUR_DIFY_API_KEY'` 处
- **不改代码**：浏览器控制台执行 `localStorage.setItem('jysb_key','app-你的Key')`

上传数据（JSONL/JSON/CSV/TXT/XLSX）→ 点「🚀 开始检测」→ 查看报告/推理链/知识库；粘贴框可输入检测指令（如「只隔离不剔除」）。

### 6.（可选）运行评测

```bat
cd benchmark
set DIFY_API_KEY=你的Dify应用Key
python run_agent_eval.py --small                          :: 500 条主评测（~25 分钟，付费）
python run_agent_eval.py --file hard_bench.jsonl --sim    :: 工具∪信号（本地免费）
python run_agent_eval.py --small --limit 100 --no-tools   :: 纯 LLM 基线（付费）
```

成绩单落盘 `benchmark/results/*.json`（指标 + 逐条判定，可审计）。

### 部署验证清单

全部完成后，用一条已知投毒数据验证全链路（10 条左右，约 2 分钟）：

1. 前端上传 `demo_data/test_backdoor.jsonl` → 点「开始检测」
2. 检测完成：报告应标出后门触发样本、正常样本零误报
3. 切「💭 推理链」页：能看到 Agent 思考与工具调用记录（unicode / embedding / label-check / kb-search）
4. 切「📚 知识库」页：显示 8 个知识库文件（kb-search 通路正常）

## 评测成绩单（2026-08-15，全部可复现）

| 实验 | 纯 LLM 基线 | 工具∪信号 | Agent 端到端 |
|---|---|---|---|
| 主评测 500 条 | — | — | **F1 100.00%**（200/200，0 误报） |
| 文本攻击切片 100 条 | 88.57% | 53.52% | **100.00%** |
| 硬样本集 50 条（5 类对抗样本） | 75.00% | 100.00% | **100.00%** |

单元测试 78/78 全绿。完整实验方法与逐条判定见 `benchmark/results/`。

## 目录结构

```
净源哨兵/
├── frontend.html               # 前端（单文件 v6.2）
├── app.py                      # Gradio 前端（备选）
├── start.bat                   # 一键启动检查脚本
├── agent/system_prompt_v6.md   # System Prompt（需同步到 Dify）
├── tool_server/                # FastAPI 工具服务器（9 工具 + 78 单测）
│   ├── tools/                  # scanner / analyzer / knowledge / response
│   └── tests/
├── knowledge_base/             # 105 条结构化攻击模式知识库（8 文件）
├── benchmark/                  # 评测数据集 + 评测脚本 + 落盘成绩单
├── demo_data/                  # 演示数据、硬样本集
└── docs/                       # 架构图
```

## 常见问题

| 问题 | 处理 |
|---|---|
| 检测报「工具服务器未运行」 | `cd tool_server && python server.py`，确认 5050 端口 |
| embedding 相关报错/超时 | 确认 Ollama 在跑且 `ollama list` 里有 bge-m3 |
| Agent 不调用工具 | 检查 Dify 工具是否用 openapi_dify.yaml 导入、系统提示词是否为 v6 |
| 知识库检索为空/知识库页无文件 | 确认 `knowledge_base/` 目录与 `tool_server/` 的相对位置未被改动；Ollama 里 bge-m3 已加载 |
| 标签投毒检测不到 | 数据文件必须带 label 列（CSV/JSONL 的 label 字段）——标签投毒的毒在 label 上，没有 label 谁也看不见 |
| 前端检测结果与预期不符 | 点「🗑 清除缓存」后重试（前端用 localStorage 缓存报告） |
| 处置按钮报错 | 升级到最新代码后重启工具服务器（历史版本处置端点有容错缺陷，已修复） |

## License

[MIT](LICENSE) © 2026 Qiuyuxin（邱宇鑫）

## 声明

本项目为**防御性安全研究工具**，用于训练数据入库前的投毒检测，无任何攻击能力。知识库引用文献均来自 arXiv 等公开渠道，已在 `knowledge_base/` 中标注。
