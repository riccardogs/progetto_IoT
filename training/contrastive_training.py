import torch
import numpy as np
from losses import nt_xent_loss
import logging
from time import time
from utils.tensorboard_logger import get_tensorboard_logger
from tqdm import tqdm

logger = logging.getLogger(__name__)

def train_epoch(model, dataloader, optimizer, device, temperature, epoch, use_grad_clip=False, max_norm=1.0):
    tensorboard_logger = get_tensorboard_logger()
    """
    Trains the model for one epoch.
    """
    model.train()
    total_loss = 0.0
    start_time = time()

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Epoch {epoch+1}')
    for batch_idx, (x_i, x_j) in pbar:
        x_i = x_i.to(device)
        x_j = x_j.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass through the model
        z_i = model(x_i)
        z_j = model(x_j)
        
        # Compute NT-Xent loss
        loss = nt_xent_loss(z_i, z_j, temperature)
        loss.backward()
        
        # Gradient clipping (evita esplosione gradienti, aiuta stabilità)
        if use_grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    epoch_duration = time() - start_time
    average_loss = total_loss / len(dataloader)
    tensorboard_logger.add_scalar('Training Loss', average_loss, epoch)
    tensorboard_logger.add_scalar('Epoch Duration', epoch_duration, epoch)
    return average_loss

def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)
    logger.info("Saved best model to %s", save_path)

def compute_validation_loss(model, dataloader, device, temperature):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x_i, x_j in dataloader:
            x_i = x_i.to(device)
            x_j = x_j.to(device)

            z_i = model(x_i)
            z_j = model(x_j)

            loss = nt_xent_loss(z_i, z_j, temperature)
            total_loss += loss.item()
    average_loss = total_loss / len(dataloader)
    model.train()
    return average_loss

def train_contrastive_model(model, dataloader, optimizer, device='cuda', num_epochs=5, temperature=0.1, val_dataloader=None, 
                            check_interval=50, min_improvement=0.01, best_model_path='best_encoder.pth',
                            use_grad_clip=True, max_norm=1.0, use_warmup=True, warmup_epochs=5):
    """
    Contrastive training loop with gradient clipping and warm-up.
    """
    tensorboard_logger = get_tensorboard_logger()
    
    # Check device availability
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA is not available. Switching to CPU.")
        device = 'cpu'
    elif device == 'mps' and not torch.backends.mps.is_available():
        logger.warning("MPS is not available. Switching to CPU.")
        device = 'cpu'
    
    model.to(device)
    model.train()
    
    # LOG DELLA TEMPERATURA USATA
    logger.info(f"Starting contrastive training for {num_epochs} epochs on {device}.")
    logger.info(f"Temperature: {temperature} (valori più alti aiutano a separare classi simili come N1 e REM)")
    if use_grad_clip:
        logger.info(f"Gradient clipping enabled (max_norm={max_norm})")
    if use_warmup:
        logger.info(f"Warm-up enabled for {warmup_epochs} epochs")
    
    best_val_loss = float('inf')
    epochs_since_improvement = 0
    total_epochs = 0
    
    # Warm-up: learning rate più basso all'inizio
    base_lr = optimizer.param_groups[0]['lr']
    
    try:
        while total_epochs < num_epochs:
            # Warm-up: aumenta gradualmente il learning rate
            if use_warmup and total_epochs < warmup_epochs:
                warmup_lr = base_lr * (total_epochs + 1) / warmup_epochs
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warmup_lr
                logger.debug(f"Warm-up epoch {total_epochs+1}: lr={warmup_lr:.6f}")
            
            # Train for one epoch
            average_loss = train_epoch(model, dataloader, optimizer, device, temperature, total_epochs, 
                                       use_grad_clip, max_norm)
            logger.info(f"Epoch [{total_epochs + 1}/{num_epochs}], Training Loss: {average_loss:.4f}")
            total_epochs += 1

            # Validate the model at specified intervals
            if val_dataloader is not None and total_epochs % check_interval == 0:
                val_loss = compute_validation_loss(model, val_dataloader, device, temperature)
                logger.info(f"Validation Loss after {total_epochs} epochs: {val_loss:.4f}")
                tensorboard_logger.add_scalar('Validation Loss', val_loss, total_epochs)

                improvement = best_val_loss - val_loss
                if improvement > min_improvement:
                    best_val_loss = val_loss
                    epochs_since_improvement = 0
                    save_model(model, best_model_path)
                    logger.info(f"Improved validation loss. Model saved to {best_model_path}.")
                    tensorboard_logger.add_scalar('Best Validation Loss', best_val_loss, total_epochs)
                else:
                    epochs_since_improvement += check_interval
                    logger.info("No significant improvement in validation loss.")
                    if epochs_since_improvement >= check_interval:
                        logger.info("Early stopping due to no improvement.")
                        break
        
        # Ripristina learning rate originale (se modificato)
        if use_warmup:
            for param_group in optimizer.param_groups:
                param_group['lr'] = base_lr
                
    except Exception as e:
        logger.error(f"An error occurred during training: {e}")
        raise e

    logger.info("Contrastive training completed.")