import math
import gzip
import random
import tqdm
import numpy as np
import os
import json
import argparse
import matplotlib.pyplot as plt
from datetime import datetime

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from nGPT_pytorch import nGPT

# Parse command line arguments
parser = argparse.ArgumentParser(description='Compare standard residual with NormResidual architecture')
parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs to train')
parser.add_argument('--depth', type=int, default=18, help='Number of layers in the models')
parser.add_argument('--dim', type=int, default=512, help='Model dimension')
parser.add_argument('--heads', type=int, default=8, help='Number of attention heads')
parser.add_argument('--dim-head', type=int, default=64, help='Dimension of each attention head')
parser.add_argument('--seq-len', type=int, default=512, help='Sequence length for training')
parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--min-norm', type=float, default=1.0, help='Minimum normalizer value for NormResidual')
parser.add_argument('--scale-factor', type=float, default=0.1, help='Scale factor for normalizer in NormResidual')
parser.add_argument('--synthetic', action='store_true', help='Use synthetic data instead of enwik8')
parser.add_argument('--save-path', type=str, default='', help='Custom path to save results (default: timestamped directory)')
args = parser.parse_args()

# Constants for both models
NUM_EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
GRAD_ACCUM_EVERY = 4
LEARNING_RATE = args.lr
VALIDATE_EVERY = 100  # Match train.py validation frequency
SEQ_LEN = args.seq_len
DEPTH = args.depth

# Model parameters
DIM = args.dim
HEADS = args.heads
DIM_HEAD = args.dim_head

# Automatically detect available devices
USE_CUDA = torch.cuda.is_available()
USE_MPS = torch.backends.mps.is_available() and not USE_CUDA
USE_AMP = USE_CUDA or USE_MPS
USE_PARAMETRIZE = True  # Whether to manually update weights after each optimizer step

# Determine device
if USE_CUDA:
    device = torch.device('cuda:0')
elif USE_MPS:
    device = torch.device('mps')
else:
    device = torch.device('cpu')

print(f"Using device: {device}")
print(f"Model config: depth={DEPTH}, dim={DIM}, heads={HEADS}, dim_head={DIM_HEAD}")
print(f"NormResidual config: min_norm={args.min_norm}, scale_factor={args.scale_factor}")
print(f"Training config: epochs={NUM_EPOCHS}, seq_len={SEQ_LEN}, batch_size={BATCH_SIZE}")

# Create synthetic dataset for comparison
class SyntheticDataset(Dataset):
    def __init__(self, vocab_size=256, seq_len=128, num_samples=1000, device='cpu'):
        super().__init__()
        self.data = torch.randint(0, vocab_size, (num_samples, seq_len + 1), device=device)
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        return self.data[index]

# Real dataset from enwik8 (same as in train.py)
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

def cycle(loader):
    while True:
        for data in loader:
            yield data

def train_model(model, name, train_loader, val_loader, epochs, device, use_norm_residual=False):
    """Train a model and return training metrics"""
    model = model.to(device)
    optim = Adam(model.parameters(), lr=LEARNING_RATE)
    
    # If not using parametrize, register normalizing on optimizer step
    if not USE_PARAMETRIZE:
        model.register_step_post_hook(optim)
    
    # Initialize gradient scaler for AMP
    # MPS doesn't support AMP, so we only use it for CUDA
    use_amp = USE_AMP and 'cuda' in str(device)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    # Metrics
    metrics = {
        'name': name,
        'train_loss': [],
        'val_loss': [],
        'epochs': []
    }
    
    if use_norm_residual:
        metrics['normalizer_values'] = []
    
    # Training loop
    for epoch in tqdm.tqdm(range(epochs), desc=f"Training {name}"):
        model.train()
        
        # Training batch
        train_losses = []
        for _ in range(GRAD_ACCUM_EVERY):
            data = next(train_loader)
            
            # Handle different devices and precision settings
            if use_amp:
                # Use autocast for CUDA 
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    loss = model(data, return_loss=True, return_breakdown=True)
                    
                    if isinstance(loss, tuple):
                        total_loss, _ = loss
                        train_losses.append(total_loss.item())
                        loss = total_loss / GRAD_ACCUM_EVERY
                    else:
                        train_losses.append(loss.item())
                        loss = loss / GRAD_ACCUM_EVERY
                    
                # Use scaler for backward pass
                scaler.scale(loss).backward()
            else:
                # Standard training for CPU or MPS
                loss = model(data, return_loss=True, return_breakdown=True)
                
                if isinstance(loss, tuple):
                    total_loss, _ = loss
                    train_losses.append(total_loss.item())
                    loss = total_loss / GRAD_ACCUM_EVERY
                else:
                    train_losses.append(loss.item())
                    loss = loss / GRAD_ACCUM_EVERY
                
                loss.backward()
        
        # Update weights based on device
        if use_amp:
            scaler.step(optim)
            scaler.update()
        else:
            optim.step()
            
        optim.zero_grad()
        
        # Record average training loss
        avg_train_loss = sum(train_losses) / len(train_losses)
        metrics['train_loss'].append(avg_train_loss)
        
        # Validation
        if epoch % VALIDATE_EVERY == 0:
            model.eval()
            with torch.no_grad():
                val_data = next(val_loader)
                loss = model(val_data, return_loss=True, return_breakdown=True)
                
                if isinstance(loss, tuple):
                    total_loss, _ = loss
                    val_loss = total_loss.item()
                else:
                    val_loss = loss.item()
                
                metrics['val_loss'].append(val_loss)
                metrics['epochs'].append(epoch)
                
                # Record normalizer values if applicable
                if use_norm_residual and hasattr(model, 'normalizer_values'):
                    metrics['normalizer_values'].append(model.normalizer_values.cpu().tolist())
                
                print(f"{name} - Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}")
    
    return metrics

def create_model(use_norm_residual=False, min_norm=1.0, scale_factor=0.1):
    """Create a model with specified parameters"""
    return nGPT(
        num_tokens=256,
        dim=DIM,
        depth=DEPTH,
        dim_head=DIM_HEAD,
        heads=HEADS,
        tied_embedding=True,
        add_value_residual=True,
        attn_norm_qk=False,  # Match train.py model configuration
        manual_norm_weights=not USE_PARAMETRIZE,
        use_norm_residual=use_norm_residual,
        norm_residual_min_norm=min_norm,
        norm_residual_scale_factor=scale_factor
    )

def plot_comparison(standard_metrics, norm_metrics, save_path='comparison_results'):
    """Plot and save comparison of training metrics"""
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(f"{save_path}/models", exist_ok=True)
    
    # Plot training loss
    plt.figure(figsize=(12, 6))
    plt.plot(standard_metrics['train_loss'], label=f"{standard_metrics['name']} - Training Loss")
    plt.plot(norm_metrics['train_loss'], label=f"{norm_metrics['name']} - Training Loss")
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_path}/training_loss_comparison.png")
    
    # Plot validation loss
    plt.figure(figsize=(12, 6))
    plt.plot(standard_metrics['epochs'], standard_metrics['val_loss'], 'o-', label=f"{standard_metrics['name']} - Validation Loss")
    plt.plot(norm_metrics['epochs'], norm_metrics['val_loss'], 'o-', label=f"{norm_metrics['name']} - Validation Loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Validation Loss Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_path}/validation_loss_comparison.png")
    
    # Plot normalizer values if available
    if 'normalizer_values' in norm_metrics and norm_metrics['normalizer_values']:
        # Take the last recorded normalizer values
        final_normalizers = norm_metrics['normalizer_values'][-1]
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(final_normalizers)), final_normalizers)
        plt.xlabel('Layer')
        plt.ylabel('Normalizer Value')
        plt.title('Final Normalizer Values Across Layers')
        plt.savefig(f"{save_path}/normalizer_values.png")
        
        # If we have enough data points, plot normalizer evolution
        if len(norm_metrics['normalizer_values']) > 5:
            plt.figure(figsize=(12, 8))
            # Plot for each layer
            for layer in range(len(final_normalizers)):
                layer_values = [epoch_values[layer] for epoch_values in norm_metrics['normalizer_values']]
                plt.plot(norm_metrics['epochs'], layer_values, label=f"Layer {layer+1}")
            
            plt.xlabel('Epochs')
            plt.ylabel('Normalizer Value')
            plt.title('Normalizer Values Evolution')
            plt.legend()
            plt.savefig(f"{save_path}/normalizer_evolution.png")
    
    # Save metrics as JSON
    with open(f"{save_path}/standard_metrics.json", 'w') as f:
        json.dump(standard_metrics, f, indent=4)
    
    with open(f"{save_path}/norm_metrics.json", 'w') as f:
        json.dump(norm_metrics, f, indent=4)
    
    # Save configuration details
    config = {
        "model_params": {
            "depth": DEPTH,
            "dim": DIM,
            "heads": HEADS,
            "dim_head": DIM_HEAD,
            "seq_len": SEQ_LEN
        },
        "training_params": {
            "epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE
        },
        "norm_residual_params": {
            "min_norm": args.min_norm,
            "scale_factor": args.scale_factor
        },
        "device": str(device)
    }
    
    with open(f"{save_path}/config.json", 'w') as f:
        json.dump(config, f, indent=4)

def main():
    # Create or load datasets
    if args.synthetic:
        print("Using synthetic data...")
        train_dataset = SyntheticDataset(seq_len=SEQ_LEN, device=device)
        val_dataset = SyntheticDataset(seq_len=SEQ_LEN, device=device)
    else:
        print("Loading enwik8 data...")
        try:
            # Load enwik8 data (same as in train.py)
            with gzip.open("./data/enwik8.gz") as file:
                data = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()
                np_train, np_valid = np.split(data, [int(90e6)])
                data_train, data_val = torch.from_numpy(np_train), torch.from_numpy(np_valid)
            
            train_dataset = TextSamplerDataset(data_train, SEQ_LEN, device)
            val_dataset = TextSamplerDataset(data_val, SEQ_LEN, device)
        except Exception as e:
            print(f"Error loading enwik8 data: {e}")
            print("Falling back to synthetic data...")
            train_dataset = SyntheticDataset(seq_len=SEQ_LEN, device=device)
            val_dataset = SyntheticDataset(seq_len=SEQ_LEN, device=device)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Wrap loaders for continuous iteration
    train_loader_iter = cycle(train_loader)
    val_loader_iter = cycle(val_loader)
    
    # Create and train standard model
    print("Creating standard model...")
    standard_model = create_model(use_norm_residual=False)
    print(f"Standard model parameters: {sum(p.numel() for p in standard_model.parameters())}")
    
    print("Training standard model...")
    standard_metrics = train_model(
        standard_model, 
        "Standard Residual", 
        train_loader_iter, 
        val_loader_iter, 
        NUM_EPOCHS, 
        device
    )
    
    # Create and train model with NormResidual
    print("Creating NormResidual model...")
    norm_model = create_model(use_norm_residual=True, min_norm=args.min_norm, scale_factor=args.scale_factor)
    print(f"NormResidual model parameters: {sum(p.numel() for p in norm_model.parameters())}")
    
    print("Training NormResidual model...")
    norm_metrics = train_model(
        norm_model, 
        "NormResidual", 
        train_loader_iter, 
        val_loader_iter, 
        NUM_EPOCHS, 
        device,
        use_norm_residual=True
    )
    
    # Plot and save comparison
    if args.save_path:
        save_path = args.save_path
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"comparison_results_{timestamp}"
    
    plot_comparison(standard_metrics, norm_metrics, save_path)
    
    # Save trained models for further evaluation
    print("Saving trained models...")
    torch.save(standard_model.state_dict(), f"{save_path}/models/standard_model.pt")
    torch.save(norm_model.state_dict(), f"{save_path}/models/norm_residual_model.pt")
    
    print(f"Comparison completed. Results saved to {save_path}/")

if __name__ == "__main__":
    torch.manual_seed(42)  # For reproducibility
    np.random.seed(42)
    random.seed(42)
    main() 