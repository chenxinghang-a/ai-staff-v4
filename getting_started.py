"""LoomLLM First-Run Setup Wizard

Run this to configure your first API key and verify everything works.

Usage:
    python -m ai_staff_v4.setup
    # or
    python getting_started.py
"""
import os
import sys

# ── Supported providers & how to get a key ──
PROVIDERS = {
    "deepseek": {
        "env": "DEEPSEEK_API_KEY",
        "url": "https://platform.deepseek.com/api_keys",
        "note": "Cheapest, no proxy needed (China-friendly)",
        "free_tier": False,
    },
    "gemini": {
        "env": "GEMINI_API_KEY",
        "url": "https://aistudio.google.com/apikey",
        "note": "Free tier available, needs proxy in China",
        "free_tier": True,
    },
    "openai": {
        "env": "OPENAI_API_KEY",
        "url": "https://platform.openai.com/api-keys",
        "note": "Needs proxy in China",
        "free_tier": False,
    },
    "moonshot": {
        "env": "MOONSHOT_API_KEY",
        "url": "https://platform.moonshot.cn/console/api-keys",
        "note": "Kimi models, no proxy needed (China-friendly)",
        "free_tier": True,
    },
    "qwen": {
        "env": "DASHSCOPE_API_KEY",
        "url": "https://dashscope.console.aliyun.com/apiKey",
        "note": "Alibaba Qwen models, no proxy needed (China-friendly)",
        "free_tier": True,
    },
    "zhipu": {
        "env": "ZHIPU_API_KEY",
        "url": "https://open.bigmodel.cn/usercenter/apikeys",
        "note": "GLM-4, no proxy needed (China-friendly)",
        "free_tier": True,
    },
}

def main():
    print("=" * 55)
    print("  🦞 LoomLLM Setup Wizard")
    print("=" * 55)
    print()

    # 1. Check existing keys
    found = {}
    for pname, pinfo in PROVIDERS.items():
        val = os.environ.get(pinfo["env"], "").strip()
        if val:
            found[pname] = val

    if found:
        print(f"  ✅ Already configured: {', '.join(found.keys())}")
        print()
        print("  Testing connection...")
        try:
            from ai_staff_v4 import AIStaff
            staff = AIStaff.from_env()
            result = staff.chat("Say 'Hello from LoomLLM!' in one sentence.")
            print(f"\n  ✅ Connection OK! Response:\n  {result[:100]}")
            print("\n  You're all set! Try:")
            print("    python -m ai_staff_v4 chat \"your question\"")
            return 0
        except Exception as e:
            print(f"\n  ❌ Connection failed: {e}")
            print("  Let's reconfigure...\n")
    else:
        print("  No API keys found. Let's set one up!\n")

    # 2. Show provider options
    print("  Available providers:")
    print()
    china = [p for p, i in PROVIDERS.items() if "China-friendly" in i["note"]]
    overseas = [p for p in PROVIDERS if p not in china]

    print("  🇨🇳 China-friendly (no proxy needed):")
    for p in china:
        i = PROVIDERS[p]
        free = " [FREE tier]" if i["free_tier"] else ""
        print(f"    {p:12s} — {i['note']}{free}")
        print(f"    {'':12s}   Get key: {i['url']}")

    print()
    print("  🌍 Overseas (proxy needed in China):")
    for p in overseas:
        i = PROVIDERS[p]
        free = " [FREE tier]" if i["free_tier"] else ""
        print(f"    {p:12s} — {i['note']}{free}")
        print(f"    {'':12s}   Get key: {i['url']}")

    print()
    print("  Quick start:")
    print("    1. Get an API key from any provider above")
    print("    2. Set it as environment variable:")
    print("       Windows: set DEEPSEEK_API_KEY=your-key")
    print("       Linux:   export DEEPSEEK_API_KEY=your-key")
    print("    3. Run this wizard again to verify")
    print()
    print("  Or use quick_start directly:")
    print("    from ai_staff_v4 import AIStaff")
    print("    staff = AIStaff.quick_start('your-key', provider='deepseek')")
    print()

    # 3. Interactive key input
    if sys.stdin.isatty():
        print("  ── Or enter a key now (won't be saved, this session only) ──")
        provider = input("  Provider [deepseek]: ").strip().lower() or "deepseek"
        if provider not in PROVIDERS:
            print(f"  Unknown provider: {provider}")
            return 1
        key = input(f"  {PROVIDERS[provider]['env']}: ").strip()
        if key:
            os.environ[PROVIDERS[provider]["env"]] = key
            print(f"\n  Testing {provider}...")
            try:
                from ai_staff_v4 import AIStaff
                staff = AIStaff.from_env()
                result = staff.chat("Say hello in one sentence.")
                print(f"\n  ✅ {provider} works! Response:\n  {result[:100]}")
                print(f"\n  To make this permanent, add to your shell profile:")
                print(f"    export {PROVIDERS[provider]['env']}={key[:8]}...")
                return 0
            except Exception as e:
                print(f"\n  ❌ Failed: {e}")
                return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
