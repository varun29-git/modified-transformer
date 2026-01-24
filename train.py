import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from tqdm import tqdm
import warnings
from pathlib import Path

from config import *
from model import build_transformer
from dataset import LanguageModelDataset


def get_causal_mask(seq_len, device):
    """Create causal mask for autoregressive generation."""
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)


def get_model(vocab_size):
    """Build the transformer model."""
    model = build_transformer(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        h=H,
        N=N,
        d_ff=D_FF,
        dropout=DROPOUT
    )
    return model


def get_ds(texts):
    """Prepare train and validation datasets."""
    # Create dataset
    dataset = LanguageModelDataset(texts, SEQ_LEN)
    
    # Split into train/val
    train_ds_size = int((1 - VAL_SPLIT) * len(dataset))
    val_ds_size = len(dataset) - train_ds_size
    train_ds, val_ds = random_split(dataset, [train_ds_size, val_ds_size])
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )
    
    val_dataloader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )
    
    return train_dataloader, val_dataloader, dataset


def train_epoch(model, dataloader, optimizer, loss_fn, device, epoch, global_step, scaler=None):
    """Train for one epoch."""
    model.train()
    batch_iterator = tqdm(dataloader, desc=f"Processing Epoch {epoch:02d}")
    
    for batch in batch_iterator:
        input_ids = batch['input_ids'].to(device)  # (B, T)
        targets = batch['targets'].to(device)      # (B, T)
        
        # Create causal mask
        mask = get_causal_mask(input_ids.shape[1], device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if scaler is not None:
            with torch.amp.autocast(device_type='cuda'):
                logits = model(input_ids, mask)  # (B, T, vocab_size)
                loss = loss_fn(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(input_ids, mask)
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        # Update progress bar
        batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
        
        global_step += 1
    
    return global_step


def run_validation(model, dataloader, loss_fn, device, writer, global_step):
    """Run validation."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        batch_iterator = tqdm(dataloader, desc="Validation")
        for batch in batch_iterator:
            input_ids = batch['input_ids'].to(device)
            targets = batch['targets'].to(device)
            
            mask = get_causal_mask(input_ids.shape[1], device)
            
            logits = model(input_ids, mask)
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
            
            total_loss += loss.item()
            num_batches += 1
            
            batch_iterator.set_postfix({"val_loss": f"{loss.item():6.3f}"})
    
    avg_loss = total_loss / num_batches
    
    # Log to tensorboard
    writer.add_scalar("validation loss", avg_loss, global_step)
    writer.flush()
    
    print(f"Validation Loss: {avg_loss:.4f} | Perplexity: {torch.exp(torch.tensor(avg_loss)):.2f}")
    
    return avg_loss


def train_model(texts):
    """Main training function."""
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model folder
    Path(MODEL_FOLDER).mkdir(parents=True, exist_ok=True)
    
    # Get datasets
    train_dataloader, val_dataloader, dataset = get_ds(texts)
    vocab_size = dataset.tokenizer.n_vocab
    
    print(f"Vocabulary size: {vocab_size}")
    print(f"Training samples: {len(train_dataloader.dataset)}")
    print(f"Validation samples: {len(val_dataloader.dataset)}")
    
    # Build model
    model = get_model(vocab_size).to(device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, eps=1e-9)
    
    # Loss function
    loss_fn = nn.CrossEntropyLoss(ignore_index=dataset.pad_id, label_smoothing=0.1).to(device)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)
    
    # Mixed precision scaler
    scaler = torch.amp.GradScaler('cuda') if USE_MIXED_PRECISION and device.type == 'cuda' else None
    
    # Load checkpoint if specified
    initial_epoch = 0
    global_step = 0
    
    if PRELOAD:
        model_filename = Path(MODEL_FOLDER) / f"{PRELOAD}.pt"
        print(f"Preloading model {model_filename}")
        state = torch.load(model_filename)
        model.load_state_dict(state["model_state_dict"])
        initial_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]
        print(f"Resumed from epoch {initial_epoch}")
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(initial_epoch, EPOCHS):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print(f"{'='*50}")
        
        # Train
        global_step = train_epoch(
            model, train_dataloader, optimizer, loss_fn, 
            device, epoch, global_step, scaler
        )
        
        # Validate
        val_loss = run_validation(
            model, val_dataloader, loss_fn, 
            device, global_step
        )
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Save checkpoint
        model_filename = Path(MODEL_FOLDER) / f"checkpoint_{epoch:02d}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": global_step,
            "val_loss": val_loss
        }, model_filename)
        print(f"Checkpoint saved: {model_filename}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_filename = Path(MODEL_FOLDER) / "best_model.pt"
            torch.save(model.state_dict(), best_model_filename)
            print(f"✓ New best model saved! Val Loss: {val_loss:.4f}")
    
    print("\n" + "="*50)
    print("Training complete!")
    print("="*50)
    
    return model




if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    
    from datasets import load_dataset
    
    dataset = load_dataset("roneneldan/TinyStories", split="train")
    dataset_subset = dataset.select(range(50000))
    texts = [item['text'] for item in dataset_subset]
    
    print(f"Loaded {len(texts)} stories")
    print(f"Sample story:\n{texts[0][:200]}...\n")

    train_model(texts)