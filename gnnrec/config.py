from pathlib import Path

# 项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent

# 数据集目录
DATA_DIR = BASE_DIR / 'data'

# 模型保存目录
MODEL_DIR = BASE_DIR / 'model'
