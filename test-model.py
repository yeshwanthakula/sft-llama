import os
import time
import urllib.request

import torch

from model import Llama3Model, generate, text_to_token_ids, token_ids_to_text
from tokenizer import Llama3Tokenizer, ChatFormat, clean_text

#######################################
# Model settings


MODEL_FILE = "llama3.2-1B-base.pth"


MODEL_CONTEXT_LENGTH = 8192  # Supports up to 131_072

# Text generation settings
if "instruct" in MODEL_FILE:
    PROMPT = "What do llamas eat?"
else:
    PROMPT = "Write a Python function to calculate the factorial of a number."

MAX_NEW_TOKENS = 150
TEMPERATURE = 0.
TOP_K = 1
#######################################


###################################
# Initialize model
##################################

url = f"https://huggingface.co/rasbt/llama-3.2-from-scratch/resolve/main/{MODEL_FILE}"

wget = "https://huggingface.co/rasbt/llama-3.2-from-scratch/resolve/main/llama3.2-1B-base.pth"



if not os.path.exists(MODEL_FILE):
    print(f"Downloading {MODEL_FILE}...")
    urllib.request.urlretrieve(url, MODEL_FILE)
    print(f"Downloaded to {MODEL_FILE}")


if "1B" in MODEL_FILE:
    from model import LLAMA32_CONFIG_1B as LLAMA32_CONFIG

else:
    raise ValueError("Incorrect model file name")

LLAMA32_CONFIG["context_length"] = MODEL_CONTEXT_LENGTH

model = Llama3Model(LLAMA32_CONFIG)
print("loading weights printing...")
model.load_state_dict(torch.load(MODEL_FILE, weights_only=True ,map_location="cpu"))
print(model.state_dict().keys())

device = (
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("mps") if torch.backends.mps.is_available() else
    torch.device("cpu")
)
model.to(device)

###################################
# Initialize tokenizer
##################################
TOKENIZER_FILE = "tokenizer.model"

tokenizer = Llama3Tokenizer("tokenizer.model")

if "instruct" in MODEL_FILE:
    tokenizer = ChatFormat(tokenizer)

###################################
# Generate text
##################################

torch.manual_seed(123)

start = time.time()
print("running inference...")
token_ids = generate(
    model=model,
    idx=text_to_token_ids(PROMPT, tokenizer).to(device),
    max_new_tokens=MAX_NEW_TOKENS,
    context_size=LLAMA32_CONFIG["context_length"],
    top_k=TOP_K,
    temperature=TEMPERATURE
)

print(f"Time: {time.time() - start:.2f} sec")

if torch.cuda.is_available():
    max_mem_bytes = torch.cuda.max_memory_allocated()
    max_mem_gb = max_mem_bytes / (1024 ** 3)
    print(f"Max memory allocated: {max_mem_gb:.2f} GB")

output_text = token_ids_to_text(token_ids, tokenizer)

if "instruct" in MODEL_FILE:
    output_text = clean_text(output_text)

print("\n\nOutput text:\n\n", output_text)