import torch
import logging
from utils.tensorboard_logger import get_tensorboard_logger
import time
from tqdm import tqdm
import numpy as np

logger = logging.getLogger(__name__)

def get_custom_class_weights(device='cuda'):
    """
    Pesi personalizzati per bilanciare N1/N3 senza uccidere N2.
    """
    # Pesi più equilibrati per proteggere N2
    class_weights = torch.tensor([0.5, 1.0, 1.2, 1.0, 1.2], dtype=torch.float).to(device)
    
    class_names = ['W', 'N1', 'N2', 'N3', 'REM']
    logger.info("=" * 50)
    logger.info("CUSTOM CLASS WEIGHTS (bilanciati):")
    for i, (name, weight) in enumerate(zip(class_names, class_weights)):
        logger.info(f"  {name}: {weight:.4f}")
    logger.info("=" * 50)
    
    return class_weights

def get_class_weights_from_loader(train_loader, num_classes=5, device='cuda'):
    """
    Calcola i pesi automaticamente dal dataset (alternativa).
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    all_labels = []
    for _, labels in train_loader:
        all_labels.extend(labels.cpu().numpy())
    
    class_weights = compute_class_weight(
        'balanced',
        classes=np.arange(num_classes),
        y=all_labels
    )
    
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    
    class_names = ['W', 'N1', 'N2', 'N3', 'REM']
    logger.info("=" * 50)
    logger.info("AUTO CLASS WEIGHTS (dal dataset):")
    for i, (name, weight) in enumerate(zip(class_names, class_weights)):
        logger.info(f"  {name}: {weight:.4f}")
    logger.info("=" * 50)
    
    return class_weights

def calculate_class_accuracy(encoder, classifier, data_loader, device='cuda', num_classes=5):
    """
    Calcola l'accuracy per singola classe.
    """
    encoder.eval()
    classifier.eval()
    
    correct_per_class = torch.zeros(num_classes, device=device)
    total_per_class = torch.zeros(num_classes, device=device)
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            embeddings = encoder(inputs)
            outputs = classifier(embeddings)
            _, predictions = torch.max(outputs, 1)
            
            for i in range(num_classes):
                mask = (labels == i)
                total_per_class[i] += mask.sum().item()
                correct_per_class[i] += (predictions[mask] == labels[mask]).sum().item()
    
    accuracy_per_class = torch.zeros(num_classes)
    for i in range(num_classes):
        if total_per_class[i] > 0:
            accuracy_per_class[i] = 100.0 * correct_per_class[i] / total_per_class[i]
    
    return accuracy_per_class.cpu().numpy()

def evaluate_classifier(encoder, classifier, data_loader, criterion, device='cuda'):
    """
    Evaluates the classifier on a given dataset.
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
    criterion=None,
    optimizer=None,
    num_epochs=50,
    device='mps',
    save_path='best_classifier/best_classifier_default.pth',
    check_interval=25,
    min_improvement=0.01,
    use_weighted_loss=True,
    custom_weights=True  # NUOVO: usa pesi personalizzati invece di auto
):
    """
    Trains the classifier while keeping the encoder frozen.
    """
    try:
        tensorboard_logger = get_tensorboard_logger()
        encoder.eval()
        classifier.to(device)
        
        # CALCOLA PESI PER CLASSE
        if use_weighted_loss and criterion is None:
            if custom_weights:
                class_weights = get_custom_class_weights(device)
            else:
                class_weights = get_class_weights_from_loader(train_loader, num_classes=5, device=device)
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
            logger.info("✅ Using weighted CrossEntropyLoss with CUSTOM weights")
        elif criterion is None:
            criterion = torch.nn.CrossEntropyLoss()
            logger.info("Using standard CrossEntropyLoss")
        
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
            
            # Calcola accuracy per classe
            class_acc = calculate_class_accuracy(encoder, classifier, val_loader, device, num_classes=5)
            logger.info(f"📊 Class accuracy - W: {class_acc[0]:.2f}%, N1: {class_acc[1]:.2f}%, N2: {class_acc[2]:.2f}%, N3: {class_acc[3]:.2f}%, REM: {class_acc[4]:.2f}%")
            
            # Log metrics
            tensorboard_logger.add_scalar('Validation/Loss', val_loss, total_epochs)
            tensorboard_logger.add_scalar('Validation/Accuracy', val_accuracy, total_epochs)
            for i, acc in enumerate(class_acc):
                tensorboard_logger.add_scalar(f'Validation/Acc_Class_{i}', acc, total_epochs)
            
            improvement = best_val_loss - val_loss
            if improvement > min_improvement:
                best_val_loss = val_loss
                best_accuracy = val_accuracy
                epochs_since_improvement = 0
                save_model(classifier, save_path)
                logger.info("✅ Improved validation loss. Model saved to %s.", save_path)
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
