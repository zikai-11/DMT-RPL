import os
import os
import sys

# 将上级目录添加到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 现在你可以导入 'configs.template' 模块
from configs.template import get_config as default_config


def get_config():
    config = default_config()

    config.result_prefix = 'results/individual_llama2'

    config.tokenizer_paths = ["../llm_model/llama-2"]
    config.model_paths = ["../llm_model/llama-2"]
    config.conversation_templates = ['llama-2']

    return config