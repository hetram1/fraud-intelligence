from pathlib import Path
import platform
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    print("=" * 60)
    print("FRAUD INTELLIGENCE PLATFORM")
    print("=" * 60)

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Python       : {sys.version.split()[0]}")
    print(f"Platform     : {platform.system()} {platform.machine()}")

    required_dirs = [
        "data/raw",
        "data/processed",
        "data/external",
        "src/data",
        "src/features",
        "src/models",
        "src/graph",
        "src/rag",
        "src/agents",
        "src/router",
        "src/evaluation",
        "src/api",
        "tests",
        "configs",
        "docker",
        "docs",
    ]

    print("\nDirectory check:")

    all_present = True

    for directory in required_dirs:
        path = PROJECT_ROOT / directory
        exists = path.is_dir()
        status = "OK" if exists else "MISSING"
        print(f"  [{status}] {directory}")

        if not exists:
            all_present = False

    print()

    if all_present:
        print("PROJECT HEALTH CHECK: PASS")
    else:
        print("PROJECT HEALTH CHECK: FAIL")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
