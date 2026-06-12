"""支持 `python -m ai_coding` 方式运行."""

from ai_coding.cli import main

if __name__ == "__main__":
    import sys
    sys.exit(main())
