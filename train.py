import math
import gzip
import random
import tqdm
import numpy as np
from contextlib import nullcontext
import os
import argparse
import json
from datetime import datetime

import torch
from torch.optim import Adam
from torch import Tensor
from torch.amp import GradScaler
from torch.utils.data import DataLoader, Dataset
import torch.nn.utils.parametrize as parametrize

from nGPT_pytorch import nGPT

# constants

NUM_BATCHES = int(1e4)
BATCH_SIZE = 4
GRAD_ACCUM_EVERY = 4
LEARNING_RATE = 1e-3
VALIDATE_EVERY = 100
SAVE_EVERY = 1000
PRIME_LENGTH = 128
GENERATE_EVERY = 500
GENERATE_LENGTH = 512
SEQ_LEN = 512

# Automatically detect available devices and enable AMP
USE_CUDA = torch.cuda.is_available()
USE_MPS = torch.backends.mps.is_available() and not USE_CUDA  # Only use MPS if CUDA is not available
USE_AMP = USE_CUDA or USE_MPS  # Enable AMP for both CUDA and MPS
USE_PARAMETRIZE = True  # whether to manually update weights after each optimizer step

# Ensure we're using a single device
if USE_CUDA:
    if torch.cuda.device_count() > 1:
        print(f"Multiple GPUs detected ({torch.cuda.device_count()}). Using first GPU.")
    device = torch.device('cuda:0')
elif USE_MPS:
    device = torch.device('mps')
else:
    device = torch.device('cpu')

print(f"Using device: {device}")

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Train the nGPT model')
parser.add_argument('--checkpoint', type=str, help='Path to checkpoint file to resume training from')
parser.add_argument('--use-norm-residual', action='store_true', help='Use the NormResidual connection architecture')
parser.add_argument('--min-norm', type=float, default=1.0, help='Minimum value for the normalizer in NormResidual')
parser.add_argument('--scale-factor', type=float, default=1.0, help='How quickly the normalizer grows in NormResidual')
args = parser.parse_args()

# helpers

def exists(v):
    return v is not None

def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return "".join(list(map(decode_token, tokens)))

# sampling helpers

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1, keepdim = True):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim, keepdim = keepdim)

# min_p
# https://arxiv.org/abs/2407.01082

def min_p_filter(logits, min_p = 0.1):
    probs = logits.softmax(dim = -1)
    max_probs = probs.amax(dim = -1, keepdim = True)
    limit = min_p * max_probs
    return torch.where(probs < limit, float('-inf'), logits)

def base_decoding(
    net,
    prompt: Tensor,
    seq_len: int,
    temperature = 1.5,
    min_p = 1e-1,
    filter_thres = 0.9,
):
    prompt_seq_len, out = prompt.shape[-1], prompt.clone()
    sample_num_times = max(0, seq_len - prompt_seq_len)

    for _ in range(sample_num_times):
        logits = net(out)
        logits = logits[:, -1]

        logits = min_p_filter(logits, min_p = min_p)
        sample = gumbel_sample(logits, temperature = temperature, dim = -1)

        out = torch.cat((out, sample), dim = -1)

    return out[..., prompt_seq_len:]

# nGPT char language model

model = nGPT(
    num_tokens = 256,
    dim = 512,
    depth = 8,
    tied_embedding = True,
    add_value_residual = True,
    attn_norm_qk = False,
    manual_norm_weights = not USE_PARAMETRIZE,
    use_norm_residual = args.use_norm_residual,
    norm_residual_min_norm = args.min_norm,
    norm_residual_scale_factor = args.scale_factor
)

# Print model architecture information
residual_type = "NormResidual" if args.use_norm_residual else "Standard Residual"
print(f"Using {residual_type} architecture")
if args.use_norm_residual:
    print(f"  - Minimum normalizer value: {args.min_norm}")
    print(f"  - Normalizer scale factor: {args.scale_factor}")

print(f"Moving model to device: {device}")
model = model.to(device)

# For MPS, ensure model is properly initialized
if str(device) == 'mps':
    # Force a forward pass to ensure all components are properly initialized
    model.train()
    dummy_input = torch.zeros(1, SEQ_LEN, dtype=torch.long, device=device)
    with torch.no_grad():
        _ = model(dummy_input)

# Disable gradient scaler for MPS as it's not supported
scaler = GradScaler(enabled = USE_AMP and not USE_MPS)

# prepare enwik8 data
with gzip.open("./data/enwik8.gz") as file:
    data = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()
    np_train, np_valid = np.split(data, [int(90e6)])
    data_train, data_val = torch.from_numpy(np_train), torch.from_numpy(np_valid)

class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len, device):
        super().__init__()
        self.data = torch.tensor(data, device=device)
        self.seq_len = seq_len

    def __len__(self):
        return self.data.size(0) // self.seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,))
        full_seq = self.data[rand_start : rand_start + self.seq_len + 1].long()
        return full_seq

train_dataset = TextSamplerDataset(data_train, SEQ_LEN, device)
val_dataset = TextSamplerDataset(data_val, SEQ_LEN, device)
train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE)
val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE)

# optimizer
optim = Adam(model.parameters(), lr = LEARNING_RATE)

train_loader = cycle(train_loader)
val_loader = cycle(val_loader)

# if not using parametrize, register normalizing on optimizer step
if not USE_PARAMETRIZE:
    model.register_step_post_hook(optim)

# Create a directory for model checkpoints if it doesn't exist
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("metrics", exist_ok=True)

# Initialize metrics tracking
metrics = {
    'train_loss': [],
    'val_loss': [],
    'epochs': [],
    'normalizer_values': []
}

# Add normalizer tracking if using NormResidual
if args.use_norm_residual:
    metrics['normalizer_values'] = []

# Function to save metrics
def save_metrics(metrics, filename="metrics/training_metrics.json"):
    with open(filename, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {filename}")

# Function to load a checkpoint
def load_checkpoint(model, optimizer, checkpoint_path):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Loaded checkpoint from epoch {start_epoch}")
        return start_epoch
    else:
        print(f"Checkpoint {checkpoint_path} not found. Starting from scratch.")
        return 0

# training
start_epoch = 0
if args.checkpoint:
    start_epoch = load_checkpoint(model, optim, args.checkpoint)
    # Try to load metrics if they exist
    metrics_file = "metrics/training_metrics.json"
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        print(f"Loaded metrics from {metrics_file}")

for i in tqdm.tqdm(range(start_epoch, NUM_BATCHES), mininterval = 10.0, desc = "training"):
    model.train()
    
    epoch_train_loss = 0.0
    num_batches = 0

    for _ in range(GRAD_ACCUM_EVERY):
        data = next(train_loader)

        # For MPS, we need to handle training differently
        if USE_MPS:
            # MPS doesn't support autocast, so we'll just run without it
            loss = model(data, return_loss = True, return_breakdown = True)
            if isinstance(loss, tuple):
                total_loss, _ = loss
                loss = total_loss / GRAD_ACCUM_EVERY
                print(f"training loss: {total_loss.item():.3f}")
                epoch_train_loss += total_loss.item()
            else:
                loss = loss / GRAD_ACCUM_EVERY
                print(f"training loss: {loss.item():.3f}")
                epoch_train_loss += loss.item()
            loss.backward()
        else:
            # Use autocast for CUDA or CPU
            with torch.autocast(device_type = 'cuda' if USE_CUDA else 'cpu', dtype = torch.float16, enabled = USE_AMP):
                loss = model(data, return_loss = True, return_breakdown = True)
                if isinstance(loss, tuple):
                    total_loss, _ = loss
                    loss = total_loss / GRAD_ACCUM_EVERY
                    print(f"training loss: {total_loss.item():.3f}")
                    epoch_train_loss += total_loss.item()
                else:
                    loss = loss / GRAD_ACCUM_EVERY
                    print(f"training loss: {loss.item():.3f}")
                    epoch_train_loss += loss.item()
            scaler.scale(loss).backward()
        
        num_batches += 1

    # Calculate average training loss for this epoch
    avg_train_loss = epoch_train_loss / num_batches
    
    if USE_MPS:
        # For MPS, we don't use the scaler
        optim.step()
    else:
        # For CUDA or CPU, use the scaler
        scaler.step(optim)
        scaler.update()

    optim.zero_grad()

    # Track validation loss
    val_loss = None
    if i % VALIDATE_EVERY == 0:
        model.eval()
        with torch.no_grad():
            valid_data = next(val_loader)
            loss = model(valid_data, return_loss = True, return_breakdown = True)
            if isinstance(loss, tuple):
                total_loss, _ = loss
                val_loss = total_loss.item()
                print(f"validation loss: {val_loss:.3f}")
            else:
                val_loss = loss.item()
                print(f"validation loss: {val_loss:.3f}")
    
    # Update metrics
    metrics['epochs'].append(i + 1)
    metrics['train_loss'].append(avg_train_loss)
    if val_loss is not None:
        metrics['val_loss'].append(val_loss)
    else:
        # If we didn't validate this epoch, use the last validation loss
        if metrics['val_loss']:
            metrics['val_loss'].append(metrics['val_loss'][-1])
        else:
            metrics['val_loss'].append(avg_train_loss)  # Fallback to training loss if no validation yet
            
    # Record normalizer values if using NormResidual
    if args.use_norm_residual and hasattr(model, 'normalizer_values'):
        metrics['normalizer_values'].append(model.normalizer_values.cpu().tolist())
    
    # Save metrics periodically
    if (i + 1) % 100 == 0:
        save_metrics(metrics)

    # Save model checkpoint every SAVE_EVERY epochs
    if (i + 1) % SAVE_EVERY == 0:
        checkpoint_path = f"checkpoints/model_epoch_{i+1}.pt"
        torch.save({
            'epoch': i + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'loss': loss.item() if not isinstance(loss, tuple) else loss[0].item(),
        }, checkpoint_path)
        print(f"Model saved to {checkpoint_path}")
        # Also save metrics with the checkpoint
        save_metrics(metrics, f"metrics/metrics_epoch_{i+1}.json")

    if i % GENERATE_EVERY == 0:
        model.eval()
        inp = random.choice(val_dataset)[:PRIME_LENGTH]
        prime = decode_tokens(inp)
        print(f"{prime} \n\n {'*' * 100}")

        prompt = inp[None, ...]
        sampled = base_decoding(model, prompt, GENERATE_LENGTH)
        base_decode_output = decode_tokens(sampled[0])
        print(f"\n\n{base_decode_output}\n")
