"""LoomLLM CLI entry point — python -m ai_staff_v4"""
import sys

def main():
    if len(sys.argv) < 2:
        print("LoomLLM v1.0.0 — The Iterative LLM Framework")
        print()
        print("Usage: python -m ai_staff_v4 <command>")
        print()
        print("Commands:")
        print("  chat <message>   — Chat with AI (zero-config)")
        print("  scan             — Scan available LLM providers")
        print("  health           — Health check all backends")
        print("  version          — Show version")
        return 0
    
    cmd = sys.argv[1].lower()
    
    if cmd == "version":
        from ai_staff_v4 import __version__
        print(f"LoomLLM v{__version__}")
        return 0
    
    if cmd == "scan":
        from ai_staff_v4.backends.smart_init import SmartInit
        registry = SmartInit.auto_configure(force_rescan=True)
        print(f"\n  Usable models: {len(registry.usable_models)}")
        for m in registry.usable_models:
            print(f"    {m.name:40s} [{m.tier:8s}] {m.provider} ({m.latency_ms:.0f}ms)")
        return 0
    
    if cmd == "health":
        from ai_staff_v4 import AIStaff
        staff = AIStaff.from_env()
        result = staff.health_check()
        for name, info in result.get("backends", {}).items():
            status = info.get("status", "unknown")
            model = info.get("model", "?")
            print(f"  {name:30s} {status:8s} {model}")
        return 0
    
    if cmd == "chat":
        from ai_staff_v4 import AIStaff
        staff = AIStaff.from_env()
        message = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "Hello!"
        result = staff.chat(message)
        print(result)
        return 0
    
    print(f"Unknown command: {cmd}")
    return 1

if __name__ == "__main__":
    sys.exit(main())
