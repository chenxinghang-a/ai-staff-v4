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
            "name": "Quick Q&A",
            "keywords": ["what is", "how to", "explain", "define", "meaning",
                        "capital of", "who is", "where is", "when was",
                        "tell me", "translate", "how many", "how much",
                        "calculate", "convert", "simple", "hello", "hi",
                        "1\\+1", "yes or no", "true or false", "list"],
            "anti_keywords": ["analyze", "design", "implement", "develop",
                            "research", "compare", "evaluate", "architect",
                            "review", "optimize", "refactor"],
            "max_length": 120,
            "experts": ["generalist"],
            "primary": "generalist",
            "needs_review": False,
            "output_format": "text",
            "max_rounds": 1,
            "followups": [],
            "desc": "Single-expert quick answer for simple Q&A and fact lookup"
        },
        "code": {
            "name": "Code Task",
            "keywords": ["write code", "implement", "function", "debug", "bug",
                        "script", "api", "algorithm", "build a", "create a",
                        "code review", "refactor", "fix the", "optimize",
                        "python", "javascript", "java", "c\\+\\+",
                        "class ", "def ", "import ", "unit test"],
            "experts": ["coder", "critic"],
            "primary": "coder",
            "needs_review": True,
            "output_format": "code",
            "max_rounds": 2,
            "followups": [
                "Check the above code for edge cases and potential bugs.",
                "Can you optimize performance or simplify the logic? Provide an improved version."
            ],
            "desc": "Code + review dual-expert pipeline for quality assurance"
        },
        "research": {
            "name": "Deep Research",
            "keywords": ["research", "analyze", "investigate", "trends",
                        "comprehensive", "in.?depth", "overview", "survey",
                        "deep.?dive", "why does", "history of", "state of",
                        "future of", "comparison", "literature"],
            "experts": ["researcher"],
            "primary": "researcher",
            "needs_review": False,
            "output_format": "markdown_report",
            "max_rounds": 4,
            "followups": [
                "Dig deeper into the most critical technical details or controversies.",
                "What important aspects or common misconceptions are easily overlooked?",
                "From a practitioner's perspective, provide an actionable guide: getting started, pitfalls, and tool recommendations."
            ],
            "desc": "Multi-round iterative deep dive for complex analysis and reports"
        },
        "decision": {
            "name": "Decision Support",
            "keywords": ["should i", "which is better", "recommend",
                        "pros and cons", "compare", "vs", "versus",
                        "choose between", "trade.?off", "worth it",
                        "alternative", "evaluate options", "decision"],
            "experts": ["planner", "researcher", "critic"],
            "primary": "planner",
            "needs_review": True,
            "output_format": "markdown_report",
            "max_rounds": 2,
            "followups": [],
            "desc": "Multi-dimensional analysis with trade-off recommendations"
        },
        "creative": {
            "name": "Creative Task",
            "keywords": ["creative", "brainstorm", "slogan", "story",
                        "write a", "design a", "come up with",
                        "headline", "tagline", "logo", "brand",
                        "marketing", "campaign", "pitch", "copy"],
            "experts": ["writer", "critic"],
            "primary": "writer",
            "needs_review": True,
            "output_format": "text",
            "max_rounds": 2,
            "followups": [
                "What can be improved or made more compelling in this proposal?",
                "Give me 3 alternative versions in different styles."
            ],
            "desc": "Creative writing + aesthetic review dual guarantee"
        },
        "collaborate": {
            "name": "Roundtable",
            "keywords": ["roundtable", "debate", "multi.?perspective",
                        "collaborate", "committee", "panel",
                        "comprehensive analysis", "cross.?domain"],
            "experts": ["planner", "researcher", "coder", "critic"],
            "primary": "planner",
            "needs_review": True,
            "output_format": "folder",
            "max_rounds": 2,
            "followups": [],
            "desc": "Multi-expert goal-driven collaboration for complex deliverables (rare)"
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
