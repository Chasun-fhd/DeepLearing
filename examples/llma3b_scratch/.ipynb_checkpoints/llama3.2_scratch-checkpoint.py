import os
import urllib.request
from pathlib import Path
from model import Llama3Model
import torch

model_base_dir = '../models'
MODEL_CONTEXT_LENGTH = 8192  # Supports up to 131_072
MAX_NEW_TOKENS = 150
TEMPERATURE = 0.
TOP_K = 1

MODEL_FILE = "llama3.2-1B-instruct.pth"
# MODEL_FILE = "llama3.2-1B-base.pth"
# MODEL_FILE = "llama3.2-3B-instruct.pth"
# MODEL_FILE = "llama3.2-3B-base.pth"
# Text generation settings
if "instruct" in MODEL_FILE:
    PROMPT = "What do llamas eat?"
else:
    PROMPT = "Llamas eat"

def download_weights():
    url = f"https://huggingface.co/rasbt/llama-3.2-from-scratch/resolve/main/{MODEL_FILE}"
    _local_model_file = Path(os.path.join(model_base_dir, MODEL_FILE))
    if not os.path.exists(_local_model_file):
        print(f"Downloading {MODEL_FILE} to {_local_model_file}")
        urllib.request.urlretrieve(url, str(_local_model_file))
        print(f"Downloaded {_local_model_file}")


def load_weights():
    # Alternatively:
    # from llms_from_scratch.llama3 import Llama3Model

    if "1B" in MODEL_FILE:
        from model import LLAMA32_CONFIG_1B as LLAMA32_CONFIG
    elif "3B" in MODEL_FILE:
        from model import LLAMA32_CONFIG_3B as LLAMA32_CONFIG
    else:
        raise ValueError("Incorrect model file name")

    LLAMA32_CONFIG["context_length"] = MODEL_CONTEXT_LENGTH

    model = Llama3Model(LLAMA32_CONFIG)
    model.load_state_dict(torch.load(MODEL_FILE, weights_only=True))

    device = (
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("mps") if torch.backends.mps.is_available() else
        torch.device("cpu")
    )
    model.to(device)
    return model

def initialize_tokenizer():
    from tokenizer import Llama3Tokenizer, ChatFormat, clean_text
    # Alternatively:
    # from llms_from_scratch.llama3 Llama3Tokenizer, ChatFormat, clean_text

    TOKENIZER_FILE = "tokenizer.model"

    url = f"https://huggingface.co/rasbt/llama-3.2-from-scratch/resolve/main/{TOKENIZER_FILE}"

    if not os.path.exists(TOKENIZER_FILE):
        urllib.request.urlretrieve(url, TOKENIZER_FILE)
        print(f"Downloaded to {TOKENIZER_FILE}")

    tokenizer = Llama3Tokenizer("tokenizer.model")

    if "instruct" in MODEL_FILE:
        tokenizer = ChatFormat(tokenizer)