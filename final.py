import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import tiktoken
import time
import os
import inspect
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import numpy as np
from tokenizer import Llama3Tokenizer
import logging
from model import generate,text_to_token_ids,token_ids_to_text
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
import functools


logging.basicConfig(
    filename='app.log',
    level=logging.INFO,  
    format='%(asctime)s - %(levelname)s - %(message)s',  
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger()
# logger.addHandler(logging.StreamHandler())


enc = Llama3Tokenizer("tokenizer.model")
LLAMA32_CONFIG_1B = {
    "vocab_size": 128_256,           # Vocabulary size
    "context_length": 8192,          # Maximum context length to use (reduced to save memory)
    "orig_context_length": 131_072,  # Context length that was used to train the model
    "emb_dim": 2048,                 # Embedding dimension
    "n_heads": 32,                   # Number of attention heads
    "n_layers": 16,                  # Number of layers
    "hidden_dim": 8192,              # Size of the intermediate dimension in FeedForward
    "n_kv_groups": 8,                # Key-Value groups for grouped-query attention
    "rope_base": 500_000.0,          # The base in RoPE's "theta"
    "dtype": torch.bfloat16,         # Lower-precision dtype to reduce memory usage
    "rope_freq": {                   # RoPE frequency scaling
        "factor": 32.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_context_length": 8192,
    }
}


def compute_rope_params(head_dim, theta_base=10_000, context_length=1024, freq_config=None, dtype=torch.float32):
    """Compute rotary position embedding parameters."""
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    # Compute the inverse frequencies
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=dtype) / head_dim))
    if freq_config is not None:
        low_freq_wavelen = freq_config["original_context_length"] / freq_config["low_freq_factor"]
        high_freq_wavelen = freq_config["original_context_length"] / freq_config["high_freq_factor"]

        wavelen = 2 * torch.pi / inv_freq

        inv_freq_llama = torch.where(
            wavelen > low_freq_wavelen, inv_freq / freq_config["factor"], inv_freq
        )

        smooth_factor = (freq_config["original_context_length"] / wavelen - freq_config["low_freq_factor"]) / (
            freq_config["high_freq_factor"] - freq_config["low_freq_factor"]
        )

        smoothed_inv_freq = (
            (1 - smooth_factor) * (inv_freq / freq_config["factor"]) + smooth_factor * inv_freq
        )

        is_medium_freq = (wavelen <= low_freq_wavelen) & (wavelen >= high_freq_wavelen)
        inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
        inv_freq = inv_freq_llama


    # Generate position indices 
    positions = torch.arange(context_length, dtype=dtype)

    # Compute the angles (m*theta)
    angles = positions[:, None] * inv_freq[None, :]  # Shape: (context_length, head_dim // 2)
    #(context_length,1) * (1,head_dim // 2)

    # Expand angles to match the head_dim
    angles = torch.cat([angles, angles], dim=1)  # Shape: (context_length, head_dim)

    # Precompute sine and cosine
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin


def apply_rope(x, cos, sin):
    """Apply rotary position embeddings to input tensor x."""
    # x: (batch_size, num_heads, seq_len, head_dim)
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"

    # Split x into first half and second half
    x1 = x[..., : head_dim // 2]  # First half
    x2 = x[..., head_dim // 2:]  # Second half

    # Adjust sin and cos shapes
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, head_dim) --> get 4 dim
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

    # Apply the rotary transformation
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)

    return x_rotated.to(dtype=x.dtype)


class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention module for Llama models with Flash Attention."""
    def __init__(self, d_in, d_out, num_heads, num_kv_groups, dtype=None):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=False, dtype=dtype)
        self.W_key = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype)
        
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups
        
        self.out_proj = nn.Linear(d_out, d_out, bias=False, dtype=dtype)
        # Mark output projection for special initialization
        self.out_proj.LLAMA_SCALE_INIT = 1

    def forward(self, x, attn_mask, cos, sin):
        b, seq_len, d_in = x.shape

        # Linear projections
        queries = self.W_query(x)  # Shape: (b, seq_len, d_out)
        keys = self.W_key(x)       # Shape: (b, seq_len, num_kv_groups * head_dim)
        values = self.W_value(x)   # Shape: (b, seq_len, num_kv_groups * head_dim)

        # Reshape queries, keys, and values
        queries = queries.view(b, seq_len, self.num_heads, self.head_dim)
        keys = keys.view(b, seq_len, self.num_kv_groups, self.head_dim)
        values = values.view(b, seq_len, self.num_kv_groups, self.head_dim)

        # Transpose for attention calculation
        queries = queries.transpose(1, 2)  # Shape: (b, num_heads, seq_len, head_dim)
        keys = keys.transpose(1, 2)        # Shape: (b, num_kv_groups, seq_len, head_dim)
        values = values.transpose(1, 2)    # Shape: (b, num_kv_groups, seq_len, head_dim)

        # Apply RoPE to queries and keys
        queries = apply_rope(queries, cos, sin)
        keys = apply_rope(keys, cos, sin)

        # Expand keys and values to match the number of heads (for grouped-query attention)
        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)

        # Use Flash Attention for better efficiency
        # The is_causal flag handles the causal masking automatically
        if hasattr(F, 'scaled_dot_product_attention'):
            # PyTorch 2.0+ has native Flash Attention
            context_vec = F.scaled_dot_product_attention(
                queries,                # (b, num_heads, seq_len, head_dim)
                keys,                   # (b, num_heads, seq_len, head_dim)
                values,                 # (b, num_heads, seq_len, head_dim)
                attn_mask=None,         # Flash Attention handles causal mask internally
                dropout_p=0.0,
                is_causal=True,
                scale=1.0 / math.sqrt(self.head_dim)
            )
        else:
            # Fallback for older PyTorch versions
            attn_scores = torch.matmul(queries, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
            attn_scores = attn_scores.masked_fill(attn_mask, float('-inf'))
            attn_weights = F.softmax(attn_scores, dim=-1)
            context_vec = torch.matmul(attn_weights, values)
        
        # Reshape back
        context_vec = context_vec.transpose(1, 2).contiguous()
        context_vec = context_vec.view(b, seq_len, self.d_out)
        
        # Output projection
        output = self.out_proj(context_vec)
        
        return output


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = nn.functional.silu(x_fc1) * x_fc2
        return self.fc3(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = GroupedQueryAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            num_kv_groups=cfg["n_kv_groups"],
            dtype=cfg["dtype"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = nn.RMSNorm(cfg["emb_dim"], eps=1e-5, dtype=cfg["dtype"])
        self.norm2 = nn.RMSNorm(cfg["emb_dim"], eps=1e-5, dtype=cfg["dtype"])

    def forward(self, x, mask, cos, sin):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x, mask, cos, sin)  # Shape [batch_size, num_tokens, emb_size]
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut  # Add the original input back

        return x


class Llama32Model(nn.Module):
    """Llama 3.2 model with GPT-2 parameters."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # Token embeddings
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])
        
        # Transformer blocks
        self.trf_blocks = nn.ModuleList([
            TransformerBlock(cfg) for _ in range(cfg["n_layers"])
        ])
        
        # Final normalization and output projection
        self.final_norm = nn.RMSNorm(cfg["emb_dim"], eps=1e-5, dtype=cfg["dtype"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Create causal attention mask (lower triangular matrix)
        self.register_buffer(
            "mask", torch.triu(torch.ones(cfg["context_length"], cfg["context_length"]), diagonal=1).bool(),
            persistent=False 
        )
        
        # Precompute rotary position embeddings
        cos, sin = compute_rope_params(
            head_dim=cfg["emb_dim"] // cfg["n_heads"],
            theta_base=cfg["rope_base"],
            context_length=cfg["context_length"],
            freq_config=cfg["rope_freq"],
            dtype=cfg["dtype"]
        )
        self.register_buffer("cos", cos,persistent=False)
        self.register_buffer("sin", sin,persistent=False)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'LLAMA_SCALE_INIT'):
                std *= (2 * self.cfg["n_layers"]) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, targets=None):
        b, seq_len = input_ids.shape
        assert seq_len <= self.cfg["context_length"], f"Input sequence length exceeds model context length"

        # Get token embeddings
        x = self.tok_emb(input_ids)
        
        # Pass through transformer blocks
        for block in self.trf_blocks:
            x = block(x, self.mask, self.cos, self.sin)
            
        # Final normalization and output projection
        x = self.final_norm(x)
        logits = self.out_head(x.to(self.cfg["dtype"]))
        
        # Calculate loss if targets are provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss
    
    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        """Configure AdamW optimizer with weight decay applied only to 2D parameters."""
        # Get parameters that require gradients
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        
        # Create optimizer groups with weight decay for 2D+ parameters only
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        
        # Print parameter counts for master process
        if ddp_is_master_process():
            num_decay_params = sum(p.numel() for p in decay_params)
            num_nodecay_params = sum(p.numel() for p in nodecay_params)
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        
        # Create AdamW optimizer with fused implementation if available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if ddp_is_master_process():
            print(f"using fused AdamW: {use_fused}")
        
        optimizer = torch.optim.AdamW(
            optim_groups, 
            lr=learning_rate, 
            betas=(0.9, 0.95), 
            eps=1e-8, 
            fused=use_fused if fused_available else False
        )
        return optimizer
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=None, eos_id=None):
        """Generate text using the model with optional temperature and top-k sampling."""
        self.eval()
        context_size = self.cfg["context_length"]
        
        for _ in range(max_new_tokens):
            # Take the last context_length tokens if sequence is too long
            idx_cond = idx[:, -context_size:]
            
            # Get logits for the next token
            logits, _ = self.forward(idx_cond)
            logits = logits[:, -1, :]  # Only use the last position
            
            # Filter logits with top_k sampling
            if top_k is not None:
                # Keep only top_k values
                top_logits, _ = torch.topk(logits, top_k)
                min_val = top_logits[:, -1]
                logits = torch.where(logits < min_val, 
                                    torch.tensor(float('-inf')).to(logits.device), 
                                    logits)
            
            # Apply temperature scaling
            if temperature > 0.0:
                logits = logits / temperature
                
                # Apply softmax to get probabilities
                probs = torch.softmax(logits, dim=-1)
                
                # Sample from the distribution
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy sampling
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
                
            if idx_next.item() == eos_id and eos_id is not None:
                # Stop generating if end-of-sequence token is encountered
                break
                
            # Append to sequence
            idx = torch.cat([idx, idx_next], dim=1)
            
        return idx


class DataLoader:
    """DataLoader for Llama model with shard support."""
    def __init__(self, B, T, process_rank=0, num_processes=1, split="train"):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.split = split
        
        # Look for shard files in the data directory
        data_root = "llama_shards"
        shards = [f for f in os.listdir(data_root) if f.startswith(f"{split}_shard_") and f.endswith(".npy")]
        shards = sorted(shards)
        self.shards = [os.path.join(data_root, s) for s in shards]
        
        assert len(shards) > 0, f"no shards found for split {split}"
        if ddp_is_master_process():
            print(f"found {len(shards)} shards for split {split}")
        
        self.reset()

    def reset(self):
        """Reset the state to start at shard zero."""
        self.current_shard = 0
        self.tokens = np.load(self.shards[self.current_shard])
        # Each process starts at a different position in the shard
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        """Get the next batch of data."""
        # print("loading the next batch...")
        B, T = self.B, self.T
        # Get a contiguous chunk of tokens
        buf = self.tokens[self.current_position:self.current_position + B * T + 1]
        
        # If we don't have enough tokens left, move to the next shard
        if len(buf) < B * T + 1:
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = np.load(self.shards[self.current_shard])
            self.current_position = self.B * self.T * self.process_rank
            # Try again with the new shard
            buf = self.tokens[self.current_position:self.current_position + B * T + 1]
        
        x = torch.tensor(buf[:-1], dtype=torch.long).view(B, T)  # inputs
        y = torch.tensor(buf[1:], dtype=torch.long).view(B, T)   # targets
        
        # Advance the position in the shard by a factor of world_size
        self.current_position += B * T * self.num_processes
        
        # If we'd go out of bounds in the next batch, prepare to move to the next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = np.load(self.shards[self.current_shard])
            self.current_position = self.B * self.T * self.process_rank
        # print("next batch loaded")
        
        return x, y


class MetricsTracker:
    """Comprehensive metrics tracking system for training monitoring."""
    
    def __init__(self, log_dir, model, save_frequency=5):
        self.log_dir = log_dir
        self.save_frequency = save_frequency
        self.metrics = defaultdict(list)
        self.layer_names = self._get_layer_names(model)
        
        # Initialize layer-wise storage
        self.layer_grad_norms = defaultdict(lambda: defaultdict(list))
        self.layer_weight_norms = defaultdict(lambda: defaultdict(list))
        
    def _get_layer_names(self, model):
        """Extract meaningful layer names from the model."""
        layer_names = []
        
        # Get the raw model (unwrap DDP if needed)
        raw_model = model.module if hasattr(model, 'module') else model
        
        # Token embedding
        layer_names.append('tok_emb')
        
        # Transformer blocks - attention and feedforward components
        for i in range(len(raw_model.trf_blocks)):
            layer_names.extend([
                f'block_{i}_att_query',
                f'block_{i}_att_key', 
                f'block_{i}_att_value',
                f'block_{i}_att_out',
                f'block_{i}_ff_w1',
                f'block_{i}_ff_w2', 
                f'block_{i}_ff_w3'
            ])
        
        # Output head
        layer_names.append('out_head')
        
        return layer_names
    
    def log_step_metrics(self, step, loss, lr, global_grad_norm, memory_usage, throughput, val_loss=None):
        """Log scalar metrics for a training step."""
        self.metrics['step'].append(step)
        self.metrics['train_loss'].append(loss)
        self.metrics['learning_rate'].append(lr)
        self.metrics['global_grad_norm'].append(global_grad_norm)
        self.metrics['memory_usage_gb'].append(memory_usage)
        self.metrics['throughput_tok_per_sec'].append(throughput)
        
        if val_loss is not None:
            self.metrics['val_loss'].append(val_loss)
    
    def log_layer_metrics(self, step, model, optimizer):
        """Log layer-wise gradient and weight norms."""
        # Get the raw model (unwrap DDP if needed)
        raw_model = model.module if hasattr(model, 'module') else model
        
        # Collect gradient norms
        grad_norms = {}
        weight_norms = {}
        
        for name, param in raw_model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                weight_norm = param.norm().item()
                
                # Simplify layer names for storage
                simplified_name = self._simplify_layer_name(name)
                grad_norms[simplified_name] = grad_norm
                weight_norms[simplified_name] = weight_norm
        
        # Store layer-wise metrics
        for layer_name in grad_norms:
            self.layer_grad_norms[step][layer_name] = grad_norms[layer_name]
            self.layer_weight_norms[step][layer_name] = weight_norms[layer_name]
    
    def _simplify_layer_name(self, full_name):
        """Convert full parameter name to simplified layer name."""
        # Handle different layer types
        if 'tok_emb' in full_name:
            return 'tok_emb'
        elif 'out_head' in full_name:
            return 'out_head'
        elif 'final_norm' in full_name:
            return 'final_norm'
        elif 'trf_blocks' in full_name:
            # Extract block number and component
            parts = full_name.split('.')
            block_idx = parts[1]  # trf_blocks.{idx}
            
            if 'att.W_query' in full_name:
                return f'block_{block_idx}_att_query'
            elif 'att.W_key' in full_name:
                return f'block_{block_idx}_att_key'
            elif 'att.W_value' in full_name:
                return f'block_{block_idx}_att_value'
            elif 'att.out_proj' in full_name:
                return f'block_{block_idx}_att_out'
            elif 'ff.w1' in full_name:
                return f'block_{block_idx}_ff_w1'
            elif 'ff.w2' in full_name:
                return f'block_{block_idx}_ff_w2'
            elif 'ff.w3' in full_name:
                return f'block_{block_idx}_ff_w3'
            elif 'norm1' in full_name:
                return f'block_{block_idx}_norm1'
            elif 'norm2' in full_name:
                return f'block_{block_idx}_norm2'
        
        return full_name  # fallback
    
    def save_metrics(self, step):
        """Save metrics to JSON file."""
        if step % self.save_frequency == 0:
            # Prepare data for JSON serialization
            save_data = {
                'scalar_metrics': dict(self.metrics),
                'layer_grad_norms': dict(self.layer_grad_norms),
                'layer_weight_norms': dict(self.layer_weight_norms),
                'last_step': step
            }
            
            # Save to JSON
            metrics_path = os.path.join(self.log_dir, 'training_metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(save_data, f, indent=2)
    
    def plot_training_dashboard(self):
        """Create individual plots for each metric type."""
        try:
            # Set style
            plt.style.use('default')
            sns.set_palette("husl")
            
            plot_configs = []
            
            # 1. Loss Curves
            if self.metrics['train_loss']:
                plot_configs.append({
                    'filename': '1_loss_curves.png',
                    'title': 'Training and Validation Loss',
                    'plot_func': self._plot_loss_curves
                })
            
            # 2. Learning Rate Schedule
            if self.metrics['learning_rate']:
                plot_configs.append({
                    'filename': '2_learning_rate.png',
                    'title': 'Learning Rate Schedule',
                    'plot_func': self._plot_learning_rate
                })
            
            # 3. Global Gradient Norm
            if self.metrics['global_grad_norm']:
                plot_configs.append({
                    'filename': '3_gradient_norm.png',
                    'title': 'Global Gradient Norm',
                    'plot_func': self._plot_gradient_norm
                })
            
            # 4. Memory Usage
            if self.metrics['memory_usage_gb']:
                plot_configs.append({
                    'filename': '4_memory_usage.png',
                    'title': 'GPU Memory Usage',
                    'plot_func': self._plot_memory_usage
                })
            
            # 5. Throughput
            if self.metrics['throughput_tok_per_sec']:
                plot_configs.append({
                    'filename': '5_throughput.png',
                    'title': 'Training Throughput',
                    'plot_func': self._plot_throughput
                })
            
            # 6. Layer-wise Gradient Norms Heatmap
            if self.layer_grad_norms:
                plot_configs.append({
                    'filename': '6_layer_gradient_norms.png',
                    'title': 'Layer-wise Gradient Norms',
                    'plot_func': lambda: self._plot_layer_heatmap_standalone(
                        self.layer_grad_norms, 'Layer-wise Gradient Norms', 'viridis'
                    )
                })
            
            # 7. Layer-wise Weight Norms Heatmap
            if self.layer_weight_norms:
                plot_configs.append({
                    'filename': '7_layer_weight_norms.png',
                    'title': 'Layer-wise Weight Norms',
                    'plot_func': lambda: self._plot_layer_heatmap_standalone(
                        self.layer_weight_norms, 'Layer-wise Weight Norms', 'plasma'
                    )
                })
            
            # Create each plot
            for config in plot_configs:
                try:
                    plt.figure(figsize=(12, 8))
                    config['plot_func']()
                    
                    plot_path = os.path.join(self.log_dir, config['filename'])
                    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    if ddp_is_master_process():
                        print(f"Saved plot: {config['filename']}")
                        
                except Exception as e:
                    if ddp_is_master_process():
                        print(f"Error creating {config['filename']}: {e}")
                    plt.close()
            
            if ddp_is_master_process():
                print(f"All training plots saved to: {self.log_dir}")
            
        except Exception as e:
            if ddp_is_master_process():
                print(f"Error creating plots: {e}")
    
    def _plot_loss_curves(self):
        """Plot training and validation loss curves."""
        steps = self.metrics['step']
        plt.plot(steps, self.metrics['train_loss'], 'b-', label='Training Loss', linewidth=2.5)
        
        if self.metrics['val_loss']:
            # Create validation steps array (val loss is logged less frequently)
            val_steps = []
            val_losses = []
            val_idx = 0
            
            for step in steps:
                if val_idx < len(self.metrics['val_loss']):
                    # Check if we have a validation loss for this step (every 5 steps typically)
                    if step % 5 == 0 or step == steps[-1]:
                        if val_idx < len(self.metrics['val_loss']):
                            val_steps.append(step)
                            val_losses.append(self.metrics['val_loss'][val_idx])
                            val_idx += 1
            
            if val_steps:
                plt.plot(val_steps, val_losses, 'r-', label='Validation Loss', 
                        linewidth=2.5, marker='o', markersize=6)
        
        plt.xlabel('Training Step', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training and Validation Loss Curves', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        # Add annotations
        if len(steps) > 1:
            initial_loss = self.metrics['train_loss'][0]
            final_loss = self.metrics['train_loss'][-1]
            improvement = ((initial_loss - final_loss) / initial_loss) * 100
            plt.text(0.02, 0.98, f'Loss Improvement: {improvement:.1f}%', 
                    transform=plt.gca().transAxes, fontsize=10, 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    verticalalignment='top')
    
    def _plot_learning_rate(self):
        """Plot learning rate schedule."""
        plt.plot(self.metrics['step'], self.metrics['learning_rate'], 
                'g-', linewidth=2.5, color='darkgreen')
        plt.xlabel('Training Step', fontsize=12)
        plt.ylabel('Learning Rate', fontsize=12)
        plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        # Add annotations for min/max LR
        max_lr = max(self.metrics['learning_rate'])
        min_lr = min(self.metrics['learning_rate'])
        plt.text(0.02, 0.98, f'Max LR: {max_lr:.2e}\nMin LR: {min_lr:.2e}', 
                transform=plt.gca().transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
                verticalalignment='top')
    
    def _plot_gradient_norm(self):
        """Plot global gradient norm."""
        plt.plot(self.metrics['step'], self.metrics['global_grad_norm'], 
                'purple', linewidth=2.5)
        plt.xlabel('Training Step', fontsize=12)
        plt.ylabel('Global Gradient Norm', fontsize=12)
        plt.title('Global Gradient Norm Over Time', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        # Add horizontal line for gradient clipping threshold (usually 1.0)
        plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Gradient Clip Threshold')
        plt.legend()
        
        # Statistics
        avg_grad_norm = np.mean(self.metrics['global_grad_norm'])
        max_grad_norm = max(self.metrics['global_grad_norm'])
        plt.text(0.02, 0.98, f'Avg: {avg_grad_norm:.3f}\nMax: {max_grad_norm:.3f}', 
                transform=plt.gca().transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='plum', alpha=0.8),
                verticalalignment='top')
    
    def _plot_memory_usage(self):
        """Plot GPU memory usage."""
        plt.plot(self.metrics['step'], self.metrics['memory_usage_gb'], 
                'orange', linewidth=2.5)
        plt.xlabel('Training Step', fontsize=12)
        plt.ylabel('GPU Memory Usage (GB)', fontsize=12)
        plt.title('GPU Memory Usage Over Time', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        avg_memory = np.mean(self.metrics['memory_usage_gb'])
        max_memory = max(self.metrics['memory_usage_gb'])
        plt.text(0.02, 0.98, f'Avg: {avg_memory:.2f} GB\nMax: {max_memory:.2f} GB\n(A100: 80GB total)', 
                transform=plt.gca().transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='moccasin', alpha=0.8),
                verticalalignment='top')
        
        # Add horizontal line for A100 total memory
        plt.axhline(y=80, color='red', linestyle='--', alpha=0.5, label='A100 Total (80GB)')
        plt.legend()
    
    def _plot_throughput(self):
        """Plot training throughput."""
        plt.plot(self.metrics['step'], self.metrics['throughput_tok_per_sec'], 
                'cyan', linewidth=2.5)
        plt.xlabel('Training Step', fontsize=12)
        plt.ylabel('Throughput (tokens/sec)', fontsize=12)
        plt.title('Training Throughput Over Time', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        avg_throughput = np.mean(self.metrics['throughput_tok_per_sec'])
        max_throughput = max(self.metrics['throughput_tok_per_sec'])
        plt.text(0.02, 0.98, f'Avg: {avg_throughput:.0f} tok/s\nMax: {max_throughput:.0f} tok/s', 
                transform=plt.gca().transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8),
                verticalalignment='top')
    
    def _plot_layer_heatmap_standalone(self, layer_data, title, cmap):
        """Create a standalone heatmap for layer-wise metrics."""
        try:
            # Convert nested dict to matrix
            steps = sorted(layer_data.keys())
            all_layers = set()
            for step_data in layer_data.values():
                all_layers.update(step_data.keys())
            
            layers = sorted(all_layers)
            
            # Create matrix: rows=layers, cols=steps
            matrix = np.full((len(layers), len(steps)), np.nan)
            
            for col_idx, step in enumerate(steps):
                for row_idx, layer in enumerate(layers):
                    if layer in layer_data[step]:
                        matrix[row_idx, col_idx] = layer_data[step][layer]
            
            # Plot heatmap
            im = plt.imshow(matrix, cmap=cmap, aspect='auto', interpolation='nearest')
            
            # Set labels
            plt.xlabel('Training Step', fontsize=12)
            plt.ylabel('Layer', fontsize=12)
            plt.title(title, fontsize=14, fontweight='bold')
            
            # Set ticks
            step_tick_interval = max(1, len(steps)//10)
            plt.xticks(range(0, len(steps), step_tick_interval),
                      [steps[i] for i in range(0, len(steps), step_tick_interval)])
            
            # Simplify layer names for y-axis
            simplified_layers = [self._simplify_layer_name_for_plot(layer) for layer in layers]
            layer_tick_interval = max(1, len(layers)//15)
            plt.yticks(range(0, len(layers), layer_tick_interval),
                      [simplified_layers[i] for i in range(0, len(layers), layer_tick_interval)], 
                      fontsize=8)
            
            # Add colorbar
            cbar = plt.colorbar(im, shrink=0.8)
            cbar.set_label('Norm Value', fontsize=10)
            
            # Add statistics text
            valid_values = matrix[~np.isnan(matrix)]
            if len(valid_values) > 0:
                plt.text(0.02, 0.98, f'Min: {np.min(valid_values):.3e}\nMax: {np.max(valid_values):.3e}\nMean: {np.mean(valid_values):.3e}', 
                        transform=plt.gca().transAxes, fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                        verticalalignment='top')
            
        except Exception as e:
            plt.text(0.5, 0.5, f'Error plotting heatmap:\n{str(e)}', 
                   transform=plt.gca().transAxes, ha='center', va='center')
            if ddp_is_master_process():
                print(f"Heatmap error: {e}")
    
    def _plot_layer_heatmap(self, ax, layer_data, title, cmap):
        """Create a heatmap for layer-wise metrics."""
        try:
            # Convert nested dict to matrix
            steps = sorted(layer_data.keys())
            all_layers = set()
            for step_data in layer_data.values():
                all_layers.update(step_data.keys())
            
            layers = sorted(all_layers)
            
            # Create matrix: rows=layers, cols=steps
            matrix = np.full((len(layers), len(steps)), np.nan)
            
            for col_idx, step in enumerate(steps):
                for row_idx, layer in enumerate(layers):
                    if layer in layer_data[step]:
                        matrix[row_idx, col_idx] = layer_data[step][layer]
            
            # Plot heatmap
            im = ax.imshow(matrix, cmap=cmap, aspect='auto', interpolation='nearest')
            
            # Set labels
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Layer')
            ax.set_title(title)
            
            # Set ticks
            ax.set_xticks(range(0, len(steps), max(1, len(steps)//5)))
            ax.set_xticklabels([steps[i] for i in range(0, len(steps), max(1, len(steps)//5))])
            
            # Simplify layer names for y-axis
            simplified_layers = [self._simplify_layer_name_for_plot(layer) for layer in layers]
            ax.set_yticks(range(0, len(layers), max(1, len(layers)//10)))
            ax.set_yticklabels([simplified_layers[i] for i in range(0, len(layers), max(1, len(layers)//10))], 
                             fontsize=8)
            
            # Add colorbar
            plt.colorbar(im, ax=ax, shrink=0.8)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error plotting heatmap:\n{str(e)}', 
                   transform=ax.transAxes, ha='center', va='center')
    
    def _simplify_layer_name_for_plot(self, layer_name):
        """Further simplify layer names for plotting."""
        if 'block_' in layer_name:
            parts = layer_name.split('_')
            if len(parts) >= 3:
                return f"B{parts[1]}-{parts[2]}"
        return layer_name[:10]  # Truncate long names


def get_lr(it, max_lr, min_lr, warmup_steps, max_steps):
    """Calculate learning rate with warmup and cosine decay."""
    # Linear warmup for warmup_steps steps
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    
    # If it > max_steps, return min learning rate
    if it > max_steps:
        return min_lr
    
    # In between, use cosine decay to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 1..0
    return min_lr + coeff * (max_lr - min_lr)


def fsdp_setup(use_fsdp=False):
    """Set up distributed training if enabled."""
    if not use_fsdp:
        return 0, 0, 1, True, "cpu"
    
    assert torch.cuda.is_available(), "FSDP requires CUDA"
    
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    
    # Set device BEFORE initializing process group
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    
    # Initialize process group with explicit device
    if not dist.is_initialized():
        init_process_group(
            backend='nccl',
            init_method='env://',
            rank=ddp_rank,
            world_size=ddp_world_size
        )
    
    # Set random seeds differently for each rank
    torch.manual_seed(1337 + ddp_rank)
    torch.cuda.manual_seed(1337 + ddp_rank)
    
    return ddp_rank, ddp_local_rank, ddp_world_size, ddp_rank == 0, device


def ddp_is_master_process():
    """Check if current process is the master process."""
    # If DDP is not initialized, consider it master
    if not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0


def ddp_cleanup():
    """Clean up distributed training resources."""
    if torch.distributed.is_initialized():
        destroy_process_group()


def train_llama(
  cfg=LLAMA32_CONFIG_1B,
    total_batch_size=524_288,      # 500K tokens (GPT-2 scale)
    micro_batch_size=4,         # Higher for cloud hardware
    seq_length=512,               # Optimal for Python functions
    max_lr=3e-4,                  # Keep GPT-2 value
    min_lr=3e-5,                  # 10x reduction
    weight_decay=0.01,             # GPT-2 standard
    grad_clip=1.0,                # Standard
    epochs=1,                     # Single pass
    tokens_per_epoch=15_000_000,  # 15M tokens total
    validation_interval=20,      # Less frequent
    save_interval=10,            # Every 10 steps
    generate_interval=20,        # Every 5 steps
    use_compile=False,             # Enable on cloud
    use_fsdp=True,                 # Use both GPUs
    log_dir="sft_log"
):
    """Train a Llama model with comprehensive metrics tracking."""
    # Auto-detect DDP if not specified
    if use_fsdp is None:
        use_fsdp = int(os.environ.get('RANK', -1)) != -1
    
    # Set up DDP or single-GPU/CPU training
    ddp_rank, ddp_local_rank, ddp_world_size, master_process, device = fsdp_setup(use_fsdp)
    
    # Determine device type for optimizer config
    device_type = "cuda" if device.startswith("cuda") else "cpu" if device == "cpu" else "mps"
    
    # Calculate steps per epoch and max steps
    assert total_batch_size % (micro_batch_size * seq_length * ddp_world_size) == 0, \
        "total_batch_size must be divisible by micro_batch_size * seq_length * ddp_world_size"
    
    grad_accum_steps = total_batch_size // (micro_batch_size * seq_length * ddp_world_size)
    steps_per_epoch = tokens_per_epoch // total_batch_size
    max_steps = steps_per_epoch * epochs
    warmup_steps = max(5, int(0.15 * max_steps))  
    
    if master_process:
        logger.info(f"Training config:")
        logger.info(f"- Total batch size: {total_batch_size} tokens")
        logger.info(f"- Micro batch size: {micro_batch_size} sequences")
        logger.info(f"- Sequence length: {seq_length} tokens")
        logger.info(f"- Gradient accumulation steps: {grad_accum_steps}")
        logger.info(f"- Steps per epoch: {steps_per_epoch}")
        logger.info(f"- Max steps: {max_steps}")
        logger.info(f"- Warmup steps: {warmup_steps}")
        logger.info(f"- Using device: {device} ({device_type})")
        logger.info(f"- Using DDP: {use_fsdp} (world size: {ddp_world_size})")
    
    # Create log directory
    if master_process:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"log.txt")
        with open(log_file, "w") as f:
            pass  # Clear the file

    # Performance optimizations
    if device_type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.backends.cuda.enable_flash_sdp(True)
        except AttributeError:
            pass
    
    torch.set_float32_matmul_precision('high')
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    # Load the base model
    model = Llama32Model(cfg)
    # if master_process:
    #     print("Loading weights...")
    # MODEL_FILE = "llama3.2-1B-base.pth"
    # model.load_state_dict(torch.load(MODEL_FILE, weights_only=True ,map_location = 'cpu'))
    if master_process:
        print("Loaded the base model weights")
        
        # Log model size
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
    # print(f"Model size: {total_params / 1e6:.2f}M parameters , moving to device {device}")
    model.to(device)
    # print(f"Model moved to device {device}")
    if master_process:
        max_mem_bytes = torch.cuda.max_memory_allocated()
        max_mem_gb = max_mem_bytes / (1024 ** 3)
        print(f"Max memory allocated: {max_mem_gb:.2f} GB")
    
    # Apply torch.compile if specified
    use_compile=False
    # print(f"Using torch.compile: {use_compile}")
    # if use_compile:
    #     model = torch.compile(model)
    
    # Wrap model in DDP if using DDP
    # print(f"Using DDP: {use_fsdp}, local rank: {ddp_local_rank}")
    if use_fsdp:
        dist.barrier()
        auto_wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=100_000_000)
        cpu_offload = CPUOffload(offload_params=True)
        model = FSDP(model, auto_wrap_policy=auto_wrap_policy, sharding_strategy=ShardingStrategy.FULL_SHARD, device_id=ddp_local_rank,cpu_offload=cpu_offload)
        dist.barrier()
    MODEL_FILE = "llama3.2-1B-base.pth"
    model.load_state_dict(torch.load(MODEL_FILE, weights_only=True ,map_location="cpu"))
    if master_process:
        print("Loaded the base model weights into FSDP model")
        logger.info(f"Base model weights loaded sucessfully")
    raw_model = model.module if use_fsdp else model

    
    # Configure optimizer
    if master_process:
        print("Configuring optimizer...")
        logger.info("Configuring optimizer...")
    optimizer = raw_model.configure_optimizers(
        weight_decay=weight_decay,
        learning_rate=max_lr,
        device_type=device_type
    )
    
    # Initialize metrics tracker
    metrics_tracker = MetricsTracker(log_dir, model, save_frequency=5) if master_process else None
    
    # Create data loaders
    train_loader = DataLoader(
        B=micro_batch_size,
        T=seq_length,
        process_rank=ddp_rank,
        num_processes=ddp_world_size,
        split="train"
    )
    val_loader = DataLoader(
        B=micro_batch_size,
        T=seq_length,
        process_rank=ddp_rank,
        num_processes=ddp_world_size,
        split="val"
    )

    try:
        # Training loop
        for step in range(max_steps):
            t0 = time.time()
            last_step = (step == max_steps - 1)
            
            # Calculate learning rate for this step
            lr = get_lr(step, max_lr, min_lr, warmup_steps, max_steps)
            if master_process:
                print(f"Step {step}: learning rate: {lr:.4e}")
                logger.info(f"Step {step}: learning rate: {lr:.4e}")
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            # Validation
            val_loss_item = None
            if step % validation_interval == 0 or last_step:
                model.eval()
                val_loader.reset()
                with torch.no_grad():
                    val_loss_accum = 0.0
                    val_loss_steps = min(10, max(1, steps_per_epoch // 10))
                    for _ in range(val_loss_steps):
                        x, y = val_loader.next_batch()
                        x, y = x.to(device), y.to(device)
                        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                            logits, loss = model(x, y)
                        loss = loss / val_loss_steps
                        val_loss_accum += loss.detach()

                if use_fsdp:
                    dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
                
                val_loss_item = val_loss_accum.item()
                
                if master_process:
                    with open(os.path.join(log_dir, "log.txt"), "a") as f:
                        f.write(f"{step} val {val_loss_item:.4f}\n")
                    
                    # Save model checkpoint
                    if step > 0 and (step % save_interval == 0 or last_step):

                        checkpoint_path = os.path.join(log_dir, f"model.pt")
                        checkpoint = {
                            'model': raw_model.state_dict(),
                            'config': raw_model.cfg,
                            'step': step,
                            'val_loss': val_loss_item,
                            'optimizer': optimizer.state_dict()
                        }
                        torch.save(checkpoint, checkpoint_path)
            
            # Text generation samples
            if (step % generate_interval == 0 or last_step) and step > 0:
                try:
                    model.eval()
                    MAX_NEW_TOKENS = 128
                    TOP_K = 1
                    
                    PROMPT = "Write a python function that calculates the factorial of a number."
                    
                    sample_rng = torch.Generator(device=device)
                    sample_rng.manual_seed(42 + ddp_rank)
                    
                    token_ids = generate(
                                model=model,
                                idx=text_to_token_ids(PROMPT, enc).to(device),
                                max_new_tokens=MAX_NEW_TOKENS,
                                context_size=LLAMA32_CONFIG_1B["context_length"],
                                top_k=TOP_K,
                                temperature=0
                            )
                    
                    if master_process or not use_fsdp:
                        generated_text = token_ids_to_text(token_ids, enc)
                        print("Generated sample:")
                        print(generated_text)
                        with open(os.path.join(log_dir, "log.txt"), "a") as f:
                            f.write(f"{step} generate {generated_text}\n")
                        
                except Exception as e:
                    if master_process:
                        print(f"Error during text generation: {e}")
                
            # Training step
            model.train()
            optimizer.zero_grad()
            loss_accum = 0.0
            
            # Gradient accumulation loop
            for micro_step in range(grad_accum_steps):
                x, y = train_loader.next_batch()
                x, y = x.to(device), y.to(device)
                
                if master_process:
                    print(f"Running forward pass for micro step {micro_step + 1}/{grad_accum_steps}...")
                    logger.info(f"Running forward pass for micro step {micro_step + 1}/{grad_accum_steps}...")
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                if master_process:
                    print("Forward pass completed")
                
                # Scale loss for gradient accumulation
                loss = loss / grad_accum_steps
                loss_accum += loss.detach()
                
                # Backward pass
                if master_process:
                    print(f"Running backward pass for micro step {micro_step + 1}/{grad_accum_steps}...")
                    logger.info(f"Running backward pass for micro step {micro_step + 1}/{grad_accum_steps}...")
                loss.backward()
            
            # Reduce loss across GPUs
            if use_fsdp:
                dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
            
            # Gradient clipping
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            # Optimizer step
            if master_process:
                print(f"Running optimizer step for step {step}...")
                logger.info(f"Runing optimizer step for step {step}")
            optimizer.step()
            
            # Synchronize CUDA for timing
            if device_type == "cuda":
                torch.cuda.synchronize()
            
            # Calculate throughput and memory usage
            t1 = time.time()
            dt = t1 - t0
            tokens_processed = micro_batch_size * seq_length * grad_accum_steps * ddp_world_size
            tokens_per_sec = tokens_processed / dt
            
            # Memory usage
            memory_usage_gb = 0.0
            if device_type == "cuda":
                memory_usage_gb = torch.cuda.max_memory_allocated() / (1024**3)
            
            # Log metrics
            if master_process and metrics_tracker:
                metrics_tracker.log_step_metrics(
                    step=step,
                    loss=loss_accum.item(),
                    lr=lr,
                    global_grad_norm=norm.item(),
                    memory_usage=memory_usage_gb,
                    throughput=tokens_per_sec,
                    val_loss=val_loss_item
                )
                
                # Log layer-wise metrics
                metrics_tracker.log_layer_metrics(step, model, optimizer)
                
                # Save metrics periodically
                metrics_tracker.save_metrics(step)
            
            # Console logging
            if master_process:
                print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr: {lr:.4e} | "
                    f"norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f} | "
                    f"mem: {memory_usage_gb:.2f}GB")
                with open(os.path.join(log_dir, "log.txt"), "a") as f:
                    f.write(f"{step} train {loss_accum.item():.6f}\n")
        
        # Final metrics save and plotting
        if master_process and metrics_tracker:
            metrics_tracker.save_metrics(max_steps - 1)
            print("Creating training dashboard...")
            metrics_tracker.plot_training_dashboard()
            print("Training completed successfully!")
        
        # Clean up DDP if used
        if use_fsdp:
            ddp_cleanup()
            
    except KeyboardInterrupt:
        if master_process:
            try:
                logger.warning(f"Training interrupted at step {step}, saving checkpoint...")
                checkpoint_path = os.path.join(log_dir, f"model_interrupted_{step:05d}.pt")
                logger.warning(f"Saving checkpoint to {checkpoint_path}")
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'config': raw_model.cfg,
                    'step': step,
                    'optimizer': optimizer.state_dict()
                }
                torch.save(checkpoint, checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")
                
                # Save metrics and create plots
                if metrics_tracker:
                    metrics_tracker.save_metrics(step)
                    metrics_tracker.plot_training_dashboard()
            except Exception as e:
                logger.error(f"Error saving checkpoint: {e}")
                    
    except Exception as e:
        if master_process:
            print(f"Training failed with error: {e}")
        if use_fsdp:
            ddp_cleanup()
        raise
    
    return model


def load_checkpoint(model, checkpoint_path, optimizer=None):
    """Load a model from a checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    step = checkpoint.get('step', 0)
    return model, optimizer, step


def main():
    train_llama()

if __name__ == "__main__":
    main()
    # export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True