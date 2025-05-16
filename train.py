"""
The main script to train the model
"""
import os
from tqdm import tqdm
import torch
from torchvision.datasets import STL10
from torch.utils.data import DataLoader
from data_transform import SimCLRDataTransform
from loss_function import NTXentLoss
from simclr_module import SimCLR
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")
torch.manual_seed(42)

# Training function
def train(model, train_loader, optimizer, criterion, epoch, epochs):
    """
    Training loop
    """
    model.train()
    total_loss = 0
    batch_count = 0
    
    with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
        for images, _ in pbar:
            x_i, x_j = images[0].to(device), images[1].to(device)
            
            optimizer.zero_grad()
            
            # Get model predictions
            _, z_i = model(x_i)
            _, z_j = model(x_j)
            
            # Compute loss
            loss = criterion(z_i, z_j)
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            # Update statistics
            total_loss += loss.item()
            batch_count += 1
            
            # Update progress bar
            pbar.set_postfix({"loss": total_loss / batch_count})
    
    return total_loss / batch_count

def main():
    """
    Main function
    """
    # Parameters
    batch_size = 128
    epochs = 100
    learning_rate = 3e-4
    weight_decay = 1e-4
    feature_dim = 128
    temperature = 0.5
    
    # Create data loaders
    transform = SimCLRDataTransform()
    train_dataset = STL10(root='./data', split='unlabeled', download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
    
    # Initialize model, optimizer, and loss
    model = SimCLR(feature_dim=feature_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = NTXentLoss(batch_size, temperature=temperature)
    
    # Create directory for saving models
    os.makedirs("models", exist_ok=True)
    
    # Training loop
    losses = []
    for epoch in range(epochs):
        loss = train(model, train_loader, optimizer, criterion, epoch, epochs)
        scheduler.step()
        losses.append(loss)
        
        # Save model checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, f"models/simclr_stl10_epoch_{epoch+1}.pt")
    
    # Save the final model
    torch.save(model.state_dict(), "models/simclr_stl10_final.pt")
    
    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title("SimCLR Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("simclr_training_loss.png")
    plt.show()
    
    return model

if __name__ == "__main__":
    # Train the SimCLR model
    model = main()