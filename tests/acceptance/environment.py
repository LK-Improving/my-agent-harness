"""显式加载本工作区.runtime私有验收环境，不显示凭据。"""

import os
import subprocess
import sys
from pathlib import Path

from dotenv import dotenv_values


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("用法：environment .runtime/验收环境文件 命令 参数...")
    root = Path.cwd().resolve()
    path = (root / sys.argv[1]).resolve()
    if not path.is_relative_to(root / ".runtime") or not path.is_file():
        raise SystemExit("只允许显式读取本工作区.runtime环境文件")
    environment = dict(os.environ)
    environment.update(
        {key: value for key, value in dotenv_values(path).items() if value is not None}
    )
    environment["PYTHONUTF8"] = "1"
    raise SystemExit(subprocess.call(sys.argv[2:], env=environment))


if __name__ == "__main__":
    main()
