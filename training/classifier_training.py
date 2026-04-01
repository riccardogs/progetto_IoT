import torch
import logging
from utils.tensorboard_logger import get_tensorboard_logger
import time
from tqdm import tqdm

logger = logging.getLogger(__name__)

def evaluate_classifier(encoder, classifier, data_loader, criterion, device='cuda'):
    """
    Evaluates the classifier on a given dataset.

    Parameters:
    - encoder (nn.Module): Frozen encoder to generate embeddings.
    - classifier (nn.Module): Classifier to evaluate.
    - data_loader (DataLoader): DataLoader providing the evaluation dataset.
    - criterion (nn.Module): Loss function.
    - device (str): Device to run the evaluation on ('cuda' or 'cpu').

    Returns:
    - avg_loss (float): Average loss over the dataset.
    - accuracy (float): Classification accuracy.

    Raises:
    - Exception: If an error occurs during evaluation.
    """
    try:
        encoder.eval()
        classifier.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                embeddings = encoder(inputs)
                outputs = classifier(embeddings)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                
                _, predictions = torch.max(outputs, 1)
                correct_predictions += (predictions == labels).sum().item()
                total_samples += labels.size(0)
        
        avg_loss = total_loss / len(data_loader)
        accuracy = correct_predictions / total_samples
        return avg_loss, accuracy
    except Exception as e:
        logger.error("Error during evaluation: %s", str(e))
        raise

def save_model(classifier, save_path):
    """
    Saves the classifier model to the specified path.

    Parameters:
    - classifier (nn.Module): Classifier model to save.
    - save_path (str): Path to save the model.

    Raises:
    - Exception: If an error occurs during saving.
    """
    try:
        torch.save(classifier.state_dict(), save_path)
        logger.info("Saved best model to %s", save_path)
    except Exception as e:
        logger.error("Error saving model: %s", str(e))
        raise



def train_epoch(encoder, classifier, train_loader, criterion, optimizer, device, epoch):
    """
    Trains the classifier for one epoch.

    Parameters:
    - encoder (nn.Module): Frozen encoder to generate embeddings.
    - classifier (nn.Module): Classifier to train.
    - train_loader (DataLoader): DataLoader for the training set.
    - criterion (nn.Module): Loss function.
    - optimizer (Optimizer): Optimizer for the classifier.
    - device (str): Device to run the training on ('cuda' or 'cpu').
    - epoch (int): Current epoch number.

    Returns:
    - avg_train_loss (float): Average training loss for the epoch.
    - epoch_duration (float): Duration of the epoch in seconds.

    Raises:
    - Exception: If an error occurs during training.
    """
    try:
        tensorboard_logger = get_tensorboard_logger()
        classifier.train()
        total_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()
        
        pbar = tqdm(train_loader, desc=f'Classifier Epoch {epoch+1}')
        
        for inputs, labels in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            with torch.no_grad():
                embeddings = encoder(inputs)
            
            optimizer.zero_grad()
            outputs = classifier(embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
        
        avg_train_loss = total_loss / len(train_loader)
        epoch_duration = time.time() - start_time
        # Log training loss and epoch duration

        tensorboard_logger.add_scalar('Train/Loss', avg_train_loss, epoch)
        tensorboard_logger.add_scalar('Train/Epoch_Duration', epoch_duration, epoch)
        return avg_train_loss, epoch_duration
    except Exception as e:
        logger.error("Error during training epoch: %s", str(e))
        raise



def train_classifier(
    encoder,
    classifier,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs=50,
    device='mps',
    save_path='best_classifier/best_classifier_default.pth',
    check_interval=25,
    min_improvement=0.01
):
    """
    Trains the classifier while keeping the encoder frozen.

    Parameters:
    - encoder (nn.Module): Frozen encoder to generate embeddings.
    - classifier (nn.Module): Classifier to train.
    - train_loader (DataLoader): DataLoader for the training set.
    - val_loader (DataLoader): DataLoader for the validation set.
    - criterion (nn.Module): Loss function.
    - optimizer (Optimizer): Optimizer for the classifier.
    - num_epochs (int): Number of training epochs.
    - device (str): Device to run the training on ('cuda' or 'cpu').
    - save_path (str): Path to save the best classifier model.
    - check_interval (int): Number of epochs between validation checks.
    - min_improvement (float): Minimum improvement in validation loss to continue training.

    Returns:
    - best_val_loss (float): Best validation loss achieved during training.

    Raises:
    - Exception: If an error occurs during training.
    """
    try:
        tensorboard_logger = get_tensorboard_logger()
        encoder.eval()
        classifier.to(device)
        best_val_loss = float('inf')
        best_accuracy = 0.0
        total_epochs = 0
        epochs_since_improvement = 0
        
        logger.info("Starting training for %d epochs", num_epochs)
        
        while total_epochs < num_epochs:
            for _ in range(check_interval):
                if total_epochs >= num_epochs:
                    break
                total_epochs += 1
                avg_train_loss, epoch_duration = train_epoch(
                    encoder, classifier, train_loader, criterion, optimizer, device, total_epochs)
                logger.info(
                    "Epoch [%d/%d], Train Loss: %.4f, Duration: %.2f sec",
                    total_epochs, num_epochs, avg_train_loss, epoch_duration
                )

            # Validate the classifier
            val_loss, val_accuracy = evaluate_classifier(encoder, classifier, val_loader, criterion, device)
            logger.info(
                "Validation Loss after %d epochs: %.4f, Validation Accuracy: %.4f",
                total_epochs, val_loss, val_accuracy
            )
            # Log validation metrics
            tensorboard_logger.add_scalar('Validation/Loss', val_loss, total_epochs)
            tensorboard_logger.add_scalar('Validation/Accuracy', val_accuracy, total_epochs)
            
            improvement = best_val_loss - val_loss
            if improvement > min_improvement:
                best_val_loss = val_loss
                best_accuracy = val_accuracy
                epochs_since_improvement = 0
                save_model(classifier, save_path)
                logger.info("Improved validation loss. Model saved to %s.", save_path)
                # Log checkpoint metrics
                tensorboard_logger.add_scalar('Checkpoint/Best_Loss', best_val_loss, total_epochs)
                tensorboard_logger.add_scalar('Checkpoint/Best_Accuracy', best_accuracy, total_epochs)
            else:
                epochs_since_improvement += check_interval
                logger.info("No significant improvement in validation loss.")
                if epochs_since_improvement >= check_interval:
                    logger.info("Early stopping due to no significant improvement.")
                    break
        
        logger.info("Training completed. Best validation loss: %.4f, Best Accuracy: %.4f", best_val_loss, best_accuracy)
        return best_val_loss
    except Exception as e:
        logger.error("Error during training: %s", str(e))
        raise