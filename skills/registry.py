"""
LoomLLM Skill Registry — Industrial automation skill integration.

Skills are knowledge packages that get injected into expert system prompts,
giving the LLM domain-specific code templates and usage patterns for tools
like pymodbus, python-snap7, OPC UA, SCADA, etc.

This is NOT a tool-calling/ReAct system — it's prompt-level integration:
  Expert has tools=[pymodbus] → SkillRegistry injects Modbus code template → LLM generates correct code
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..core.constants import PACKAGE_ROOT
from ..core.verbose import log

# Skills directory
SKILLS_DIR = PACKAGE_ROOT / "skills"


@dataclass
class SkillConfig:
    """A skill definition with code template and usage tips."""
    id: str
    name: str
    description: str
    domain_tags: list[str] = field(default_factory=list)
    python_package: str = ""
    code_template: str = ""
    tips: list[str] = field(default_factory=list)
    install: str = ""
    deploy: str = ""


class SkillRegistry:
    """Load and manage skill configurations from YAML files."""

    _skills: dict[str, SkillConfig] = {}

    @classmethod
    def load_all(cls) -> int:
        """Load all skill definitions from skills/*.yaml."""
        try:
            import yaml
        except ImportError:
            log.warn("PyYAML not installed, using built-in skills only")
            cls._load_builtin()
            return len(cls._skills)

        count = 0
        for f in SKILLS_DIR.glob("*.yaml"):
            try:
                with open(f, 'r', encoding='utf-8') as fh:
                    data = yaml.safe_load(fh)
                if isinstance(data, list):
                    for item in data:
                        skill = cls._parse(item)
                        cls._skills[skill.id] = skill
                        count += 1
            except Exception as e:
                log.warn(f"Failed to load skill {f.name}: {e}")

        if count == 0:
            cls._load_builtin()
            count = len(cls._skills)

        log.system(f"Loaded {count} skill(s)")
        return count

    @classmethod
    def _parse(cls, data: dict) -> SkillConfig:
        tips = data.get('tips', [])
        if isinstance(tips, str):
            tips = [tips]
        return SkillConfig(
            id=data.get('id', 'unknown'),
            name=data.get('name', 'Unnamed'),
            description=data.get('description', ''),
            domain_tags=data.get('domain_tags', []),
            python_package=data.get('python_package', ''),
            code_template=data.get('code_template', ''),
            tips=tips,
            install=data.get('install', ''),
            deploy=data.get('deploy', ''),
        )

    @classmethod
    def _load_builtin(cls):
        """Built-in fallback when no YAML files found."""
        cls._skills["pymodbus"] = SkillConfig(
            id="pymodbus", name="Modbus通信",
            description="Modbus TCP/RTU通信",
            python_package="pymodbus",
            domain_tags=["modbus", "plc"],
        )
        cls._skills["python-snap7"] = SkillConfig(
            id="python-snap7", name="西门子S7通信",
            description="西门子S7 PLC通信",
            python_package="python-snap7",
            domain_tags=["s7", "siemens", "plc"],
        )
        cls._skills["open62541"] = SkillConfig(
            id="open62541", name="OPC UA通信",
            description="OPC UA客户端/服务端",
            python_package="opcua-asyncio",
            domain_tags=["opcua", "industry40"],
        )
        cls._skills["fuxa-scada"] = SkillConfig(
            id="fuxa-scada", name="SCADA组态",
            description="Web端SCADA/HMI组态",
            domain_tags=["scada", "hmi"],
        )
        cls._skills["qelectrotech"] = SkillConfig(
            id="qelectrotech", name="电气CAD",
            description="电气原理图/接线图绘制",
            domain_tags=["electrical-design", "cad"],
        )
        cls._skills["pypsa"] = SkillConfig(
            id="pypsa", name="电力系统分析",
            description="潮流计算/最优潮流/电网规划",
            python_package="pypsa",
            domain_tags=["power-system", "simulation"],
        )

    @classmethod
    def get(cls, skill_id: str) -> Optional[SkillConfig]:
        if not cls._skills:
            cls.load_all()
        return cls._skills.get(skill_id)

    @classmethod
    def list_all(cls) -> list[SkillConfig]:
        if not cls._skills:
            cls.load_all()
        return list(cls._skills.values())

    @classmethod
    def search(cls, query: str) -> list[SkillConfig]:
        """Search skills by keyword matching."""
        if not cls._skills:
            cls.load_all()
        q = query.lower()
        results = []
        for skill in cls._skills.values():
            if (q in skill.name.lower() or q in skill.description.lower() or
                any(q in tag for tag in skill.domain_tags)):
                results.append(skill)
        return results

    @classmethod
    def get_context(cls, tool_ids: list[str]) -> str:
        """
        Build skill context string for injection into system prompts.

        Given a list of tool IDs (from ExpertConfig.tools), load each skill's
        code template and tips, and format them into a unified context block.

        Returns empty string if no skills found.
        """
        if not tool_ids:
            return ""

        if not cls._skills:
            cls.load_all()

        parts = []
        for tid in tool_ids:
            skill = cls._skills.get(tid)
            if not skill:
                continue

            section = [f"### {skill.name} ({tid})"]
            section.append(skill.description)

            if skill.python_package:
                section.append(f"安装: `pip install {skill.python_package}`")

            if skill.install:
                section.append(f"安装:\n{skill.install}")

            if skill.deploy:
                section.append(f"部署:\n{skill.deploy}")

            if skill.code_template:
                section.append("代码模板:")
                section.append(f"```python\n{skill.code_template.strip()}\n```")

            if skill.tips:
                section.append("要点:")
                for tip in skill.tips:
                    section.append(f"- {tip}")

            parts.append("\n".join(section))

        if not parts:
            return ""

        header = "## 可用技能工具\n以下技能已就绪，你可以在生成代码时直接使用这些库和模式：\n"
        return header + "\n\n".join(parts)


__all__ = ['SkillRegistry', 'SkillConfig', 'SKILLS_DIR']
