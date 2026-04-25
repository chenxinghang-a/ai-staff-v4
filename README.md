# 基于多智能体大模型的电气工程智能化任务处理研究

> AI-Staff V4 — Multi-Agent LLM Framework for Electrical Engineering

## 毕业设计概述

本项目研究如何利用大语言模型(LLM)多智能体协作技术，解决电气工程领域任务处理的**可靠性**和**专业性**问题。

### 核心问题

| 问题 | 传统LLM方案 | 本框架方案 |
|------|------------|-----------|
| 输出不可靠 | 单次生成，无校验 | V5质量门控闭环，多轮迭代直至合格 |
| 领域知识缺失 | 通用prompt | 6个电气专业专家 + 6个工业技能注入 |
| 任务类型单一 | 一刀切 | TaskClassifier自动路由6种执行策略 |
| 资源浪费 | 简单问题也用复杂流程 | 简单问答1次调用搞定，复杂任务才启动多专家 |
| 无工业标准 | 不考虑安全规范 | 审查专家对照GB/IEC标准逐项检查 |

### 答辩核心话术

> "我不是做了一个聊天机器人框架，我解决的是**工业场景下LLM不可靠**的问题——质量门控防幻觉，成本感知省资源，多专家协作模拟电气团队。"

## 系统架构

```
用户输入 → TaskClassifier(智能路由)
  ├── 快速问答 → 电气工程助手 (1次调用)
  ├── 控制程序设计 → 工程师+审查员 (2轮闭环, Skill注入)
  ├── 系统分析 → 电气系统分析员 (4轮迭代)
  ├── 方案决策 → 规划师+分析员+审查员 (多专家)
  ├── 技术文档 → 文档工程师+审查员 (2轮)
  └── 多专家协作 → 全团队协作 (复杂项目)
```

### 关键创新点

1. **Skill上下文注入** — 专家绑定工业技能(pymodbus/snap7/OPC UA/SCADA/CAD/PyPSA)，LLM自动获得专业代码模板，生成正确的工业通信代码
2. **V5质量门控闭环** — 审查专家对输出评分，不达标自动迭代改进，确保工业级可靠性
3. **成本感知路由** — 简单问题500倍token节省，复杂任务才启动多专家协作
4. **安全优先审查** — 方案审核专家对照GB 50054/GB 50065等标准逐项检查安全联锁

## 技术栈

- **语言**: Python 3.10+
- **LLM后端**: OpenAI兼容API (Gemini/DeepSeek/Qwen/GLM/Claude等10+)
- **通信协议**: Modbus TCP/RTU, S7, OPC UA
- **工业软件**: SCADA (FUXA), 电气CAD (QElectroTech)
- **电力仿真**: PyPSA (潮流计算/OPF)
- **零配置启动**: 自动检测API Key → 自动选择最优后端

## 6个电气专家

| 专家 | 职责 | 绑定技能 | 审查 |
|------|------|---------|------|
| 电气工程助手 | 通用问答 | — | ❌ |
| 电气系统分析员 | 故障诊断/系统分析 | PyPSA | ❌ |
| 控制程序工程师 | PLC/SCADA/通信编程 | pymodbus, snap7, OPC UA | ✅ |
| 技术文档工程师 | 文档/方案书/规程 | QElectroTech | ✅ |
| 方案审核专家 | 安全合规审查 | — | ❌ |
| 电气项目规划师 | 项目分解/施工方案 | FUXA SCADA | ✅ |

## 6个工业技能

| 技能 | 用途 | 代码模板 |
|------|------|---------|
| pymodbus | Modbus TCP/RTU通信 | ✅ |
| python-snap7 | 西门子S7 PLC通信 | ✅ |
| open62541 | OPC UA客户端/服务端 | ✅ |
| fuxa-scada | Web SCADA/HMI组态 | ✅ |
| qelectrotech | 电气原理图/接线图 | ✅ |
| pypsa | 电力系统潮流/OPF | ✅ |

## 快速开始

```bash
pip install -e .
export GEMINI_API_KEY=xxx  # 或 DEEPSEEK_API_KEY=xxx
python -m ai_staff_v4
```

```python
from ai_staff_v4 import AIStaff

staff = AIStaff.from_env()

# PLC控制逻辑（自动路由到控制程序工程师+审查员）
result = staff.chat("设计三相异步电机星三角启动PLC控制逻辑")

# Modbus通信（自动注入pymodbus代码模板）
result = staff.chat("用Modbus TCP读取PLC寄存器数据")

# 故障诊断（自动路由到电气系统分析员，4轮迭代）
result = staff.chat("10kV线路馈线保护动作跳闸，分析故障类型")
```

## 测试

```bash
python -m pytest ai_staff_v4/tests/test_core.py -v
```

## 项目结构

```
ai_staff_v4/
├── experts/          # 6个电气专家定义 + 智能路由分类器
├── skills/           # 6个工业技能(代码模板+Tips)
├── agents/           # V5协作闭环 + 执行器 + 审查员
├── backends/         # 10+ LLM后端 + 零配置SmartInit
├── core/             # 事件总线/预算/记忆/校验/日志
├── main_mod/         # AIStaff主编排器
├── examples/         # 电气场景Demo
│   ├── electrical_demo.py       # 3个典型场景
│   └── electrical_skill_demo.py # Skill集成效果
└── tests/            # 单元测试
```

## 声明

本项目使用AI辅助开发，框架设计和工程决策由作者独立完成。

## License

MIT
