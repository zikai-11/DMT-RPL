import os
import os
import sys

# 将上级目录添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 现在你可以导入 'configs.template' 模块
from configs.template import get_config as default_config

def get_config():
    
    config = default_config()
    config.model_paths = [
        "llm_model/vicuna",
        # more models
    ]
    config.tokenizer_paths = [
        "llm_model/vicuna",
        # more tokenizers
    ]
    return config