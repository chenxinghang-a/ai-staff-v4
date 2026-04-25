"""Generate thesis chapters 1, 2, 6 via LoomLLM + Gemini"""
import os, sys
sys.path.insert(0, r'c:\Users\cxx\WorkBuddy\Claw')

from ai_staff_v4 import AIStaff

staff = AIStaff.from_env()
doc_dir = r'c:\Users\cxx\WorkBuddy\Claw\ai_staff_v4\docs'

def gen(prompt, filename):
    result = staff.chat(prompt, return_details=True)
    text = result.final_text if hasattr(result, 'final_text') else str(result)
    path = os.path.join(doc_dir, filename)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f'{filename}: {len(text)} chars')

gen(
    '你是电气工程毕设论文作者。撰写第一章绪论，4000字。'
    '论文题目：面向电气工程智能化任务处理的多智能体大模型协作系统研究与实现。'
    '包含：1.1研究背景与意义（工业4.0+大模型+电气工程痛点：幻觉/不合规/无专业上下文）'
    '1.2国内外研究现状（单LLM在工业领域局限、多智能体系统研究进展、知识注入方法对比）'
    '1.3研究内容与目标（6专家路由+6Skill注入+V5质量门控+成本感知）'
    '1.4论文组织结构。学术论文风格，第三人称，引用[1]-[10]，章末给参考文献。',
    '第一章_绪论.md'
)

gen(
    '你是电气工程毕设论文作者。撰写第二章相关技术与理论基础，5000字。'
    '论文题目：面向电气工程智能化任务处理的多智能体大模型协作系统研究与实现。'
    '包含：2.1大语言模型技术基础（Transformer架构/指令微调/上下文学习）'
    '2.2多智能体系统理论（MAS基本概念/协作与竞争机制/共识协议）'
    '2.3提示词工程与上下文学习（Zero-shot/Few-shot/CoT/ReAct，对比本系统的Prompt级知识注入）'
    '2.4工业自动化技术基础（PLC/SCADA/Modbus/OPC UA/S7协议/电力系统分析）'
    '2.5本章小结。学术论文风格，第三人称，引用文献，章末给参考文献。',
    '第二章_相关技术.md'
)

gen(
    '你是电气工程毕设论文作者。撰写第六章总结与展望，2000字。'
    '论文题目：面向电气工程智能化任务处理的多智能体大模型协作系统研究与实现。'
    '包含：6.1研究工作总结（4个创新点：Skill上下文注入、V5质量门控、成本感知路由、工业安全审查）'
    '6.2不足与展望（4个方向：实时数据集成、多模态输入、领域微调、生产环境部署）。'
    '学术论文风格，第三人称，客观严谨。',
    '第六章_总结与展望.md'
)

print('ALL DONE')
