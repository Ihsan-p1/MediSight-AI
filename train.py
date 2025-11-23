import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import time
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from models import EmotionNet, DrowsinessNet, PainNet
from dataset_loader import get_dataloaders

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cpu', save_path="best_model.pth"):
    best_acc = 0.0
    scaler = GradScaler()
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0
            total_samples = 0

            # Progress bar
            loop = tqdm(dataloader, desc=phase, leave=False)

            for inputs, labels in loop:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    # AMP Context
                    with autocast():
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                    if phase == 'train':
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total_samples += labels.size(0)
                
                # Update progress bar
                loop.set_postfix(loss=loss.item())

            epoch_loss = running_loss / total_samples
            epoch_acc = running_corrects.double() / total_samples
            
            print(f"{phase} Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

            # Deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), save_path)
                print(f"🔥 New best saved -> {save_path}")

    print(f"\nTraining done. Best val acc: {best_acc:.4f}")
    return best_acc

def test_model(model, test_loader, device="cuda"):
    print("\nStarting Testing...")
    model.eval()
    total = 0
    correct = 0

    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Testing"):
            x = x.to(device)
            y = y.to(device)

            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    acc = correct / total
    print(f"Test Accuracy: {acc:.4f}")
    return acc

def main():
    parser = argparse.ArgumentParser(description="Train MediSight-AI Models (Pro Version)")
    parser.add_argument("--model", type=str, required=True, choices=["emotion", "fatigue", "pain"], help="Model to train")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--dry_run", action="store_true", help="Run for 1 epoch with small subset for testing")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Data
    print(f"Loading data for {args.model}...")
    # Adjust workers for Windows if needed (0 is safest for debugging, but user asked for 4)
    # We will trust the user's request for 4, but if it fails, we know why.
    train_loader, val_loader, test_loader, num_classes = get_dataloaders(
        args.model, 
        batch_size=args.batch_size, 
        num_workers=args.workers, 
        pin_memory=True
    )
    
    # Initialize Model
    print(f"Initializing {args.model} model with {num_classes} classes...")
    if args.model == "emotion":
        model = EmotionNet(num_classes=num_classes)
    elif args.model == "fatigue":
        model = DrowsinessNet(num_classes=num_classes)
    elif args.model == "pain":
        model = PainNet(num_classes=num_classes)
        
    model = model.to(device)
    
    # Setup Training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    os.makedirs("checkpoints", exist_ok=True)
    save_path = os.path.join("checkpoints", f"{args.model}_best.pth")
    
    epochs = 1 if args.dry_run else args.epochs
    
    # Train
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=epochs, device=device, save_path=save_path)
    
    # Load best model for testing
    print(f"\nLoading best model from {save_path} for testing...")
    model.load_state_dict(torch.load(save_path))
    
    # Test
    test_model(model, test_loader, device=device)

if __name__ == "__main__":
    main()
