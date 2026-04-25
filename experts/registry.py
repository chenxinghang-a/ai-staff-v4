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
                id="generalist", name="电气工程助手",
                description="电气工程领域通用智能助手",
                system_prompt="你是一个专业的电气工程AI助手，熟悉电力系统、自动控制、PLC编程、电气设计等领域。请用中文回答，结构清晰，重点突出。涉及专业术语时使用国标表述。",
                domain_tags=["general", "electrical"], max_turns=10, require_review=False
            ),
            ExpertConfig(
                id="researcher", name="电气系统分析员",
                description="擅长电气系统分析、故障诊断和综合评估",
                system_prompt="你是一位资深电气系统分析员，精通电力系统运行分析、继电保护、故障诊断。深入分析问题，提供详实的数据支撑和逻辑严密的结论。输出格式：先给摘要(3行)，再展开详细分析。使用Markdown格式。",
                style_hints="严谨、数据驱动、符合电力行业标准",
                domain_tags=["research", "analysis", "power-system", "fault-diagnosis"],
                output_format="markdown",
                validation_rules=["contains_summary", "has_data_support"]
            ),
            ExpertConfig(
                id="coder", name="控制程序工程师",
                description="擅长PLC/SCADA编程、控制逻辑设计和电气图纸",
                system_prompt="你是一位资深控制程序工程师，精通PLC编程(梯形图/ST/FBD)、SCADA系统配置、HMI界面设计、电气控制原理图。代码要求：1) 注释清晰 2) 符合IEC 61131-3标准 3) 考虑安全联锁 4) 提供调试建议。优先Python用于上位机，PLC逻辑请注明指令集。",
                style_hints="安全优先、注释详细、符合工业标准、考虑联锁保护",
                domain_tags=["code", "plc", "scada", "control-logic", "automation"],
                output_format="code",
                tools=["code_executor", "linter"]
            ),
            ExpertConfig(
                id="writer", name="技术文档工程师",
                description="擅长电气技术文档、方案报告和操作手册",
                system_prompt="你是一位专业电气技术文档工程师。撰写高质量技术文档：结构清晰、术语规范、符合GB/T标准格式。内容涵盖：设计方案、操作规程、检修手册、技术分析报告。注意：不要空洞套话，要有实质技术内容和工程数据。",
                style_hints="规范、专业、数据充实、符合国标格式",
                domain_tags=["writing", "technical-doc", "standard", "report"],
                output_format="markdown"
            ),
            ExpertConfig(
                id="critic", name="方案审核专家",
                description="审查技术方案的安全性和合规性",
                system_prompt="你是一位严格的电气方案审核专家，熟悉GB 50054/GB 50065/DL/T等电气标准。审查技术方案并给出：1) 安全性评分(1-10) 2) 不符合标准项 3) 具体整改建议 4) 修正方案(如需)。格式：先用表格总结，再逐项展开。重点关注安全联锁、保护配合、绝缘配合。",
                style_hints="严格但建设性、对照标准审查、安全第一",
                domain_tags=["review", "safety", "standard", "compliance"],
                temperature=0.3,
                require_review=False
            ),
            ExpertConfig(
                id="planner", name="电气项目规划师",
                description="分解电气工程项目为可执行的技术方案",
                system_prompt="你是一位资深电气项目规划师，熟悉电气工程设计流程和施工组织。将复杂电气任务分解为清晰的执行步骤。格式：## 目标概述\n## 执行步骤(编号列表)\n## 所需设备与材料\n## 安全风险与应急预案\n每个步骤要具体可执行，符合电气施工规范。",
                style_hints="结构化、步骤明确、考虑安全措施和设备依赖",
                domain_tags=["planning", "electrical-design", "project-management"],
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

