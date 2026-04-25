"""
LoomLLM 电气工程 Skill 集成示例
================================
演示Skill上下文注入如何让LLM生成专业的工业通信代码：
1. Modbus通信 — coder专家自动获得pymodbus代码模板
2. S7 PLC通信 — coder专家自动获得snap7代码模板
3. 潮流计算 — researcher专家自动获得pypsa代码模板
"""
import os
from ai_staff_v4 import AIStaff

# 自动从环境变量检测API配置
staff = AIStaff.from_env()


# ═══════════════════════════════════════════════════
#  示例1: Modbus通信程序（coder → 自动注入pymodbus模板）
# ═══════════════════════════════════════════════════
print("=" * 60)
print("示例1: Modbus通信程序 — Skill上下文自动注入")
print("=" * 60)

result1 = staff.chat(
    "用Python写一个Modbus TCP客户端，连接192.168.1.100的PLC，"
    "读取地址100开始的10个保持寄存器，并把地址200的寄存器写入数值100",
    return_details=True
)
print(f"模式: {result1.strategy_mode} | 专家: {result1.experts_used}")
print(f"质量: {result1.quality_score}/10 | 轮次: {result1.rounds_used}")
print(f"输出预览:\n{result1.final_text[:300]}...")
print()


# ═══════════════════════════════════════════════════
#  示例2: S7 PLC数据采集（coder → 自动注入snap7模板）
# ═══════════════════════════════════════════════════
print("=" * 60)
print("示例2: S7 PLC数据采集 — Skill上下文自动注入")
print("=" * 60)

result2 = staff.chat(
    "写一个Python脚本连接S7-1200 PLC(192.168.1.10)，"
    "读取DB1.DBD0开始的浮点数(温度值)和DB1.DBX4.0的布尔量(泵状态)，"
    "每2秒采集一次并打印",
    return_details=True
)
print(f"模式: {result2.strategy_mode} | 专家: {result2.experts_used}")
print(f"质量: {result2.quality_score}/10 | 轮次: {result2.rounds_used}")
print(f"输出预览:\n{result2.final_text[:300]}...")
print()


# ═══════════════════════════════════════════════════
#  示例3: 电力系统潮流计算（researcher → 自动注入pypsa模板）
# ═══════════════════════════════════════════════════
print("=" * 60)
print("示例3: 潮流计算 — Skill上下文自动注入")
print("=" * 60)

result3 = staff.chat(
    "用Python实现一个简单电力系统的潮流计算："
    "2个母线(20kV和0.4kV)，1台变压器(10MVA)，1台发电机(100MW)，1个负荷(50MW)，"
    "求线路潮流分布",
    return_details=True
)
print(f"模式: {result3.strategy_mode} | 专家: {result3.experts_used}")
print(f"质量: {result3.quality_score}/10 | 轮次: {result3.rounds_used}")
print(f"输出预览:\n{result3.final_text[:300]}...")
print()


# ═══════════════════════════════════════════════════
#  成本统计
# ═══════════════════════════════════════════════════
print("=" * 60)
print("Skill集成效果对比")
print("=" * 60)
for name, r in [("Modbus通信", result1), ("S7采集", result2), ("潮流计算", result3)]:
    print(f"  {name}: {r.total_tokens} tokens | {r.rounds_used}轮 | "
          f"模式={r.strategy_mode} | 质量={r.quality_score}/10")
