"""
LoomLLM 电气工程应用示例
==============================
演示多智能体LLM框架在电气自动化场景下的3种典型应用：
1. PLC控制逻辑生成+安全审查
2. 电力系统故障诊断分析
3. 电气设计方案对比决策
"""
import os

# ── 环境配置 ──
# 支持 Gemini / DeepSeek / OpenAI 等任何OpenAI兼容API
# 方式1: 环境变量
#   export GEMINI_API_KEY=xxx  或  export DEEPSEEK_API_KEY=xxx
# 方式2: 代码直接传入
#   staff = AIStaff(api_key="xxx", base_url="https://api.deepseek.com")

from ai_staff_v4 import AIStaff

# 自动从环境变量检测API配置
staff = AIStaff.from_env()


# ═══════════════════════════════════════════════════
#  示例1: PLC控制逻辑生成（控制程序设计模式）
#  TaskClassifier自动识别为code任务 → 触发工程师+审查员闭环
# ═══════════════════════════════════════════════════
print("=" * 60)
print("示例1: PLC控制逻辑生成")
print("=" * 60)

result1 = staff.chat(
    "设计一个三相异步电机星三角启动的PLC控制逻辑，"
    "要求：1) 星形启动延时3秒后切换三角形运行 "
    "2) 包含过载保护和紧急停止联锁 "
    "3) 用梯形图逻辑描述",
    return_details=True
)
print(f"模式: {result1.strategy_mode}")
print(f"质量评分: {result1.quality_score}")
print(f"迭代次数: {result1.rounds_used}")
print(f"输出: {list(result1.deliverables.values())[0][:200]}...")
print()


# ═══════════════════════════════════════════════════
#  示例2: 电力系统故障诊断（系统分析模式）
#  TaskClassifier自动识别为research任务 → 分析员多轮迭代
# ═══════════════════════════════════════════════════
print("=" * 60)
print("示例2: 电力系统故障诊断")
print("=" * 60)

result2 = staff.chat(
    "某10kV线路馈线保护动作跳闸，故障录波显示A相电流突增3倍，"
    "零序电流为0，试分析可能的故障类型和定位方法",
    return_details=True
)
print(f"模式: {result2.strategy_mode}")
print(f"迭代次数: {result2.rounds_used}")
print(f"输出: {list(result2.deliverables.values())[0][:200]}...")
print()


# ═══════════════════════════════════════════════════
#  示例3: 电气设计方案对比（方案决策模式）
#  TaskClassifier自动识别为decision任务 → 规划师+分析员+审查员
# ═══════════════════════════════════════════════════
print("=" * 60)
print("示例3: 电气设计方案对比")
print("=" * 60)

result3 = staff.chat(
    "某工厂配电系统改造，方案A采用传统接触器控制，方案B采用智能断路器+Modbus通信，"
    "请从可靠性、成本、可维护性三个维度对比分析",
    return_details=True
)
print(f"模式: {result3.strategy_mode}")
print(f"质量评分: {result3.quality_score}")
print(f"输出: {list(result3.deliverables.values())[0][:200]}...")
print()


# ═══════════════════════════════════════════════════
#  成本统计
# ═══════════════════════════════════════════════════
print("=" * 60)
print("成本统计")
print("=" * 60)
for name, r in [("PLC逻辑", result1), ("故障诊断", result2), ("方案对比", result3)]:
    print(f"  {name}: {r.total_tokens} tokens | {r.rounds_used}轮 | 模式={r.strategy_mode}")
