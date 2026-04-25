from __future__ import annotations
import json, os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Optional

# Intra-package imports
from ..core.constants import PACKAGE_ROOT
from ..core.verbose import log

# Expert configs directory
EXPERTS_DIR = PACKAGE_ROOT / "experts"

@dataclass
class ExpertConfig:
    """An expert role definition."""
    id: str
    name: str
    description: str
    system_prompt: str
    style_hints: str = ""          # Tone, format, language preferences
    domain_tags: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)     # Tool names this expert can use
    max_turns: int = 5             # Max conversation turns
    temperature: float = 0.7
    model_override: str = ""       # Use specific model for this expert
    api_profile: str = ""          # Use specific API backend (multi-mode only)
    
    # Quality gates
    require_review: bool = True
    output_format: str = "text"    # text | json | markdown | code
    validation_rules: list[str] = field(default_factory=list)

class ExpertRegistry:
    """Load and manage expert configurations from YAML files."""
    
    _experts: dict[str, ExpertConfig] = {}
    
    @classmethod
    def load_all(cls) -> int:
        """Load all .yaml files from experts directory."""
        try:
            import yaml
        except ImportError:
            log.warn("PyYAML not installed, using built-in experts only")
            cls._load_builtin()
            return len(cls._experts)
        
        count = 0
        for f in EXPERTS_DIR.glob("*.yaml"):
            try:
                with open(f, 'r', encoding='utf-8') as fh:
                    data = yaml.safe_load(fh)
                if isinstance(data, list):
                    for item in data:
                        exp = cls._parse(item)
                        cls._experts[exp.id] = exp
                        count += 1
                elif isinstance(data, dict):
                    exp = cls._parse(data)
                    cls._experts[exp.id] = exp
                    count += 1
            except Exception as e:
                log.warn(f"Failed to load {f.name}: {e}")
        
        if count == 0:
            cls._load_builtin()
            count = len(cls._experts)
        
        return count
    
    @classmethod
    def _parse(cls, data: dict) -> ExpertConfig:
        return ExpertConfig(
            id=data.get('id', 'unknown'),
            name=data.get('name', 'Unnamed'),
            description=data.get('description', ''),
            system_prompt=data.get('system_prompt', 'You are a helpful assistant.'),
            style_hints=data.get('style_hints', ''),
            domain_tags=data.get('domain_tags', []),
            tools=data.get('tools', []),
            max_turns=data.get('max_turns', 5),
            temperature=data.get('temperature', 0.7),
            model_override=data.get('model_override', ''),
        api_profile=data.get('api_profile', ''),
            require_review=data.get('require_review', True),
            output_format=data.get('output_format', 'text'),
            validation_rules=data.get('validation_rules', [])
        )
    
    @classmethod
    def _load_builtin(cls):
        """Built-in default experts when no YAML files found."""
        builtin_experts = [
            ExpertConfig(
                id="generalist", name="General Assistant",
                description="Versatile assistant for most tasks",
                system_prompt="You are a professional, efficient, and friendly AI assistant. Answer clearly with well-structured responses. Highlight key points.",
                domain_tags=["general"], max_turns=10, require_review=False
            ),
            ExpertConfig(
                id="researcher", name="Research Analyst",
                description="Expert in information gathering, analysis, and synthesis",
                system_prompt="You are a senior research analyst. Provide in-depth analysis with solid data support and rigorous logic. Format: 3-line summary first, then detailed analysis in Markdown.",
                style_hints="Rigorous, data-driven, multi-perspective analysis",
                domain_tags=["research", "analysis", "report"],
                output_format="markdown",
                validation_rules=["contains_summary", "has_data_support"]
            ),
            ExpertConfig(
                id="coder", name="Senior Engineer",
                description="Expert in coding, debugging, and architecture design",
                system_prompt="You are a senior software engineer. Code requirements: 1) Clear comments 2) Proper error handling 3) Follow best practices 4) Include usage examples. Prefer Python unless another language is specified.",
                style_hints="Detailed comments, examples, edge case handling",
                domain_tags=["code", "programming", "debugging", "architecture"],
                output_format="code",
                tools=["code_executor", "linter"]
            ),
            ExpertConfig(
                id="writer", name="Content Creator",
                description="Expert in copywriting and creative content",
                system_prompt="You are a professional content creator. Produce high-quality content: compelling titles, clear structure, engaging language. Avoid empty platitudes — deliver substance and unique perspectives.",
                style_hints="Vivid, opinionated, avoids cliches",
                domain_tags=["writing", "content", "creative", "copywriting"],
                output_format="markdown"
            ),
            ExpertConfig(
                id="critic", name="Quality Reviewer",
                description="Reviews output quality and suggests improvements",
                system_prompt="You are a strict quality reviewer. For the input, provide: 1) Quality score (1-10) 2) Key issues list 3) Specific improvement suggestions 4) Revised version (if needed). Format: summary table first, then item-by-item breakdown.",
                style_hints="Strict but constructive, specific not vague",
                domain_tags=["review", "quality", "critique"],
                temperature=0.3,  # Lower temp for consistent reviews
                require_review=False
            ),
            ExpertConfig(
                id="planner", name="Project Planner",
                description="Breaks down complex tasks into actionable step plans",
                system_prompt="You are a senior project planner. Break complex tasks into clear, ordered execution steps. Format: ## Goal Overview\n## Execution Steps (numbered)\n## Required Resources\n## Risks & Alternatives\nMake each step specific and actionable.",
                style_hints="Structured, clear steps, considers dependencies",
                domain_tags=["planning", "breakdown", "architecture"],
                require_review=True,
                output_format="markdown"
            )
        ]
        for exp in builtin_experts:
            cls._experts[exp.id] = exp
    
    @classmethod
    def get(cls, expert_id: str) -> Optional[ExpertConfig]:
        if not cls._experts:
            cls.load_all()
        return cls._experts.get(expert_id)
    
    @classmethod
    def list_all(cls) -> list[ExpertConfig]:
        return list(cls._experts.values())
    
    @classmethod
    def search(cls, query: str) -> list[ExpertConfig]:
        """Search experts by keyword matching."""
        q = query.lower()
        results = []
        for exp in cls._experts.values():
            if (q in exp.name.lower() or q in exp.description.lower() or
                any(q in tag for tag in exp.domain_tags)):
                results.append(exp)
        return results
    
    @classmethod
    def create_expert_file(cls, expert: ExpertConfig, filename: str = None):
        """Save an expert config to YAML file."""
        try:
            import yaml
        except ImportError:
            log.warn("Need pyyaml to save expert files")
            return
        
        fname = filename or f"{expert.id}.yaml"
        filepath = EXPERTS_DIR / fname
        data = {
            'id': expert.id, 'name': expert.name,
            'description': expert.description,
            'system_prompt': expert.system_prompt,
            'style_hints': expert.style_hints,
            'domain_tags': expert.domain_tags,
            'tools': expert.tools, 'max_turns': expert.max_turns,
            'temperature': expert.temperature,
            'model_override': expert.model_override,
            'require_review': expert.require_review,
            'output_format': expert.output_format,
            'validation_rules': expert.validation_rules
        }
        with open(filepath, 'w', encoding='utf-8') as fh:
            yaml.dump(data, fh, allow_unicode=True, default_flow_style=False)
            log.success(f"Saved expert '{expert.id}' to {filepath}")


__all__ = ['ExpertConfig', 'ExpertRegistry', 'EXPERTS_DIR']

