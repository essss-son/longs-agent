import asyncio
import sys

from .app import main, trace_cli

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "trace":
        trace_cli(sys.argv[2:])
    else:
        asyncio.run(main())
