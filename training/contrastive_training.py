import torch
import numpy as np
from losses import nt_xent_loss
import logging
from time import time
from utils.tensorboard_logger import get_tensorboard_logger
from tqdm import tqdm

logger = logging.getLogger(__name__)

def train_epoch(model, dataloader, optimizer, device, temperature, epoch):
    tensorboard_logger = get_tensorboard_logger()
    """
    Trains the model for one epoch.

    Parameters:
    - model (torch.nn.Module): The neural network model to train.
    - dataloader (torch.utils.data.DataLoader): DataLoader providing contrastive pairs.
    - optimizer (torch.optim.Optimizer): Optimizer for updating the model's parameters.
    - device (str): Device to run the training on ('cuda' or 'cpu').
    - temperature (float): Temperature parameter for NT-Xent loss.
    - epoch (int): Current epoch number.

    Returns:
        float: Average loss for the epoch.
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
        loss.backward()  # Backpropagation
        optimizer.step()  # Update model parameters
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    epoch_duration = time() - start_time
    average_loss = total_loss / len(dataloader)
    tensorboard_logger.add_scalar('Training Loss', average_loss, epoch)
    tensorboard_logger.add_scalar('Epoch Duration', epoch_duration, epoch)
    return average_loss

def save_model(model, save_path):
    """
    Saves the model to the specified path.

    Parameters:
    - model (torch.nn.Module): The model to save.
    - save_path (str): Path to save the model.

    Raises:
        Exception: If an error occurs during saving.
    """
    torch.save(model.state_dict(), save_path)
    logger.info("Saved best model to %s", save_path)

def compute_validation_loss(model, dataloader, device, temperature):
    """
    Compute the average loss on the validation dataset.
    """
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
                            check_interval=50, min_improvement=0.01, best_model_path='best_encoder.pth'):
    """
    Contrastive training loop for a model using NT-Xent loss.

    Parameters:
    - model (torch.nn.Module): The neural network model to train.
    - dataloader (torch.utils.data.DataLoader): DataLoader providing contrastive pairs.
    - optimizer (torch.optim.Optimizer): Optimizer for updating the model's parameters.
    - device (str): Device to run the training on ('cuda' or 'cpu'). Default is 'cuda'.
    - num_epochs (int): Number of training epochs. Default is 5.
    - temperature (float): Temperature parameter for NT-Xent loss. Default is 0.1.
    - val_dataloader (torch.utils.data.DataLoader, optional): DataLoader for validation data. Default is None.
    - check_interval (int): Number of epochs between validation checks. Default is 50.
    - min_improvement (float): Minimum improvement in validation loss to continue training. Default is 0.01.
    - best_model_path (str): Path to save the best model. Default is 'best_encoder.pth'.

    Raises:
    - Exception: If an error occurs during training.
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
    logger.info(f"Starting contrastive training for {num_epochs} epochs on {device}.")
    
    best_val_loss = float('inf')
    epochs_since_improvement = 0
    total_epochs = 0

    try:
        while total_epochs < num_epochs:
            # Train for one epoch
            average_loss = train_epoch(model, dataloader, optimizer, device, temperature, total_epochs)
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
    except Exception as e:
        logger.error(f"An error occurred during training: {e}")
        raise e

    logger.info("Contrastive training completed.")