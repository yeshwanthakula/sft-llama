import json
import os
import math
import numpy as np
from tokenizer import Llama3Tokenizer

def create_shards_for_ddp(input, output_dir, num_shards=8):
    """
    Create tokenized shards for distributed training.
    Using 8 shards (4 per GPU) gives flexibility during training.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize tokenizer - using GPT-2 tokenizer for compatibility with Llama
    enc = Llama3Tokenizer("tokenizer.model")
    
    # Load all examples
    with open(input, 'r', encoding='utf-8') as f:
        examples = f.read()
    

    all_tokens = enc.encode(examples)
    
    # Calculate examples per shard
    tokens_per_shard = math.ceil(len(all_tokens) / num_shards)

    
    # Track total tokens
    total_tokens = 0
    val_tokens = []
    
    # Create shards
    for shard_idx in range(num_shards):
        start_idx = shard_idx * tokens_per_shard
        end_idx = min((shard_idx + 1) * tokens_per_shard, len(all_tokens))
        
        # Extract tokens for this shard
        shard_tokens = all_tokens[start_idx:end_idx]
        split_point = int(0.95 * len(shard_tokens))
        train_tokens = shard_tokens[:split_point]
        val_tokens.extend(shard_tokens[split_point:]) 
        
        # Save as numpy array
        shard_array = np.array(train_tokens, dtype=np.int32)
        shard_path = os.path.join(output_dir, f"train_shard_{shard_idx:02d}.npy")
        np.save(shard_path, shard_array)
        
        print(f"Created shard {shard_idx}: {len(train_tokens)} tokens")
    

  
    
    val_path = os.path.join(output_dir, "val_shard_00.npy")
    np.save(val_path, np.array(val_tokens, dtype=np.int32))
    
    print(f"Created validation shard: {len(val_tokens)} tokens")
    print(f"Total tokens across all shards: {total_tokens}")
    print(f"Average tokens per shard: {total_tokens / num_shards}")

# Usage
# create_shards_for_ddp('opencodeinstruct_15m_robust.txt', 'llama_shards')