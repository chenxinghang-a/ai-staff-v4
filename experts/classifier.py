from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TaskStrategy:
    """Optimal execution strategy for a detected task type."""
    mode: str                    # direct | code | research | collaborate | analysis
    display_name: str            # Human-readable name
    experts: list[str]           # Expert IDs to use
    primary_expert: str          # Main expert for single-expert modes
    needs_review: bool           # Whether critic should check output
    output_format: str           # text | code | markdown_report | folder
    max_rounds: int              # Max conversation rounds
    auto_followups: list[str]     # Auto-generated follow-up questions
    description: str             # Why this strategy was chosen


class TaskClassifier:
    """
    V3 CORE: Automatically classify any user input into an optimal execution strategy.
    
    Key principle (from user feedback):
    - NOT everything needs a roundtable meeting!
    - Simple questions → direct answer (1 call, 1 expert)
    - Code tasks → coder + critic (2 experts, focused)
    - Research → deep iterative dive (1 expert, multiple rounds)
    - Complex decisions → multi-expert collaboration (only when needed!)
    
    This is THE differentiator: users just say what they want,
    the framework figures out HOW to do it optimally.
    """

    # Task type definitions with keyword patterns
    TASK_DEFINITIONS: dict[str, dict] = {
        "direct": {
            "name": "快速问答",
            "keywords": ["什么是", "怎么", "如何", "解释", "定义", "意思", "1+1",
                        "hello", "hi", "你好", "谢谢", "翻译", "convert",
                        "多少", "几", "谁", "哪里", "什么时候", "list",
                        "what is", "how to", "explain", "define",
                        "额定", "参数", "规格"],
            "anti_keywords": ["分析", "设计", "实现", "开发", "研究", "对比",
                            "评估", "方案", "架构", "系统",
                            "analyze", "design", "implement"],
            "max_length": 120,
            "experts": ["generalist"],
            "primary": "generalist",
            "needs_review": False,
            "output_format": "text",
            "max_rounds": 1,
            "followups": [],
            "desc": "单专家快速回答，适合简单问答和参数查询"
        },
        "code": {
            "name": "控制程序设计",
            "keywords": ["写代码", "实现", "function", "debug", "bug", "程序",
                        "PLC", "梯形图", "SCADA", "HMI", "控制逻辑",
                        "算法", "script", "api", "接口", "函数",
                        "编程", "python", "上位机", "通信",
                        "Modbus", "S7", "OPC UA", "寄存器", "DB块",
                        "write code", "implement", "build a"],
            "anti_keywords": ["对比", "比较", "vs", "选型", "方案选择"],
            "experts": ["coder", "critic"],
            "primary": "coder",
            "needs_review": True,
            "output_format": "code",
            "max_rounds": 2,
            "followups": [
                "请检查以上控制逻辑的安全联锁和边界条件。",
                "能否优化性能或简化逻辑？给出符合IEC标准的改进版本。"
            ],
            "desc": "控制程序+安全审查双专家流程，确保工业可靠性"
        },
        "research": {
            "name": "系统分析",
            "keywords": ["研究", "分析", "调研", "综述", "趋势", "原理",
                        "为什么", "故障诊断", "运行分析", "现状", "前景",
                        "深入", "详细分析", "全面", "报告",
                        "research", "analyze", "investigate"],
            "experts": ["researcher"],
            "primary": "researcher",
            "needs_review": False,
            "output_format": "markdown_report",
            "max_rounds": 4,
            "followups": [
                "基于以上分析，进一步深挖关键技术细节或安全隐患。",
                "有哪些容易被忽视的重要方面或常见认知误区？",
                "从工程实践角度，给出具体行动指南：调试步骤、避坑建议、工具推荐。"
            ],
            "desc": "电气系统分析员多轮迭代，适合故障分析和系统评估"
        },
        "decision": {
            "name": "方案决策",
            "keywords": ["应该", "选择", "建议", "哪个好", "买哪个",
                        "推荐", "优缺点", "比较", "对比", "选型",
                        "是否值得", "or", "vs", "取舍",
                        "通信方案", "方案选择", "技术路线",
                        "should i", "which is better", "recommend"],
            "experts": ["planner", "researcher", "critic"],
            "primary": "planner",
            "needs_review": True,
            "output_format": "markdown_report",
            "max_rounds": 2,
            "followups": [],
            "desc": "多维度技术方案对比+安全合规审查"
        },
        "creative": {
            "name": "技术文档",
            "keywords": ["文档", "报告", "方案书", "操作手册", "说明书",
                        "规程", "编写", "起草", "标书", "技术规格",
                        "原理图", "接线图", "配电", "电气图纸",
                        "write a", "design a", "写一份", "起草一份"],
            "experts": ["writer", "critic"],
            "primary": "writer",
            "needs_review": True,
            "output_format": "text",
            "max_rounds": 2,
            "followups": [
                "这个技术文档还有什么可以完善的地方？",
                "给出3个不同侧重点的替代版本。"
            ],
            "desc": "技术文档+标准合规审查双重保障"
        },
        "collaborate": {
            "name": "多专家协作",
            "keywords": ["圆桌", "讨论", "综合", "多方", "协作",
                        "全流程", "端到端"],
            "experts": ["planner", "researcher", "coder", "critic"],
            "primary": "planner",
            "needs_review": True,
            "output_format": "folder",
            "max_rounds": 2,
            "followups": [],
            "desc": "多专家目标驱动协作，产出完整技术方案（仅复杂项目触发）"
        },
    }

    def __init__(self):
        self._definitions = self.TASK_DEFINITIONS
        self._sync_experts()

    def _sync_experts(self):
        """从ExpertRegistry同步可用专家，覆盖硬编码的experts列表"""
        try:
            from .registry import ExpertRegistry
            if not ExpertRegistry._experts:
                ExpertRegistry.load_all()
            available = {e.id for e in ExpertRegistry.list_all()}
        except Exception:
            return  # registry不可用时用默认值

        for task_type, cfg in self._definitions.items():
            # 过滤掉不存在的专家，fallback到generalist
            valid = [e for e in cfg["experts"] if e in available]
            if not valid:
                valid = ["generalist"] if "generalist" in available else []
            cfg["experts"] = valid
            cfg["primary"] = valid[0] if valid else "generalist"

    def classify(self, user_input: str) -> TaskStrategy:
        """
        Classify user input into optimal execution strategy.
        
        Uses: keyword matching + length heuristics + anti-keyword filtering.
        
        Returns a TaskStrategy that tells ai-staff EXACTLY what to do:
        - Which experts to use
        - How many rounds
        - What output format
        - Whether to review
        
        The caller (auto_run) just executes this strategy blindly.
        """
        text_lower = user_input.lower().strip()
        text_len = len(user_input)

        scores: dict[str, float] = {}

        for task_type, config in self._definitions.items():
            score = 0.0

            # Keyword matching (positive signals)
            kw_score = sum(2.0 for kw in config["keywords"] if re.search(kw, text_lower, re.IGNORECASE))
            score += kw_score

            # Anti-keyword penalty (this is NOT this type of task)
            anti_kws = config.get("anti_keywords", [])
            anti_penalty = sum(3.0 for akw in anti_kws if akw in text_lower)
            score -= anti_penalty

            # Length heuristic: short queries favor "direct"
            if "max_length" in config and text_len <= config["max_length"]:
                score += 1.5
            elif task_type == "direct" and text_len > 100:
                score -= 1.0  # Long queries are probably not simple Q&A

            # Multi-sentence / complex structure suggests research or decision
            sentence_count = text_lower.count('.') + text_lower.count('?') + text_lower.count('\n')
            if task_type in ("research", "collaborate") and sentence_count >= 2:
                score += 1.5

            # Question mark suggests direct Q&A or decision
            if '?' in user_input or '？' in user_input:
                if task_type == "direct":
                    score += 1.0
                elif task_type == "decision":
                    score += 0.8

            scores[task_type] = score

        # Find best match
        best_type = max(scores.keys(), key=lambda k: scores[k]) if scores else "direct"
        best_score = scores[best_type]

        # Fallback: if no clear signal, use direct for short, research for long
        if best_score <= 0:
            best_type = "direct" if text_len < 50 else "research"

        cfg = self._definitions[best_type]

        return TaskStrategy(
            mode=best_type,
            display_name=cfg["name"],
            experts=cfg["experts"],
            primary_expert=cfg["primary"],
            needs_review=cfg["needs_review"],
            output_format=cfg["output_format"],
            max_rounds=cfg["max_rounds"],
            auto_followups=cfg["followups"],
            description=cfg["desc"]
        )

    def explain(self, user_input: str, strategy: TaskStrategy = None) -> str:
        """Explain why this strategy was chosen (for transparency)."""
        if not strategy:
            strategy = self.classify(user_input)
        return (
            f"[{strategy.display_name}] ({strategy.mode})\n"
            f"  Strategy: {strategy.description}\n"
            f"  Experts: {', '.join(strategy.experts)}\n"
            f"  Rounds: {strategy.max_rounds} | Review: {'yes' if strategy.needs_review else 'no'}\n"
            f"  Output: {strategy.output_format}"
        )


__all__ = ['TaskStrategy', 'TaskClassifier']
