import os

# 对于Conda环境，检查CONDA_PREFIX环境变量
conda_env_path = os.getenv('CONDA_PREFIX')

if conda_env_path:
    print(f"当前Conda虚拟环境位于: {conda_env_path}")
else:
    print("可能不在Conda虚拟环境中")

import torch

print(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
