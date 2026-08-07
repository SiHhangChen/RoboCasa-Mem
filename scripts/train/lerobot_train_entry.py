#!/usr/bin/env python
"""入口脚本: 供 accelerate launch 使用 (accelerate 不支持 -m 模块方式)

用法:
    accelerate launch scripts/train/lerobot_train_entry.py --policy.type=groot ...
"""

from lerobot.scripts.lerobot_train import main

if __name__ == "__main__":
    main()
