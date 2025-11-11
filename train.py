"""
The main script to train the model
"""
import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from dataset_and_transform import SimCLRDataTransform, DAVISDataset
from loss_function import NTXentLoss
from simclr_module import SimCLR
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")
torch.manual_seed(42)

def simclr_collate_fn(batch):
    """
    Custom collate function for SimCLR that handles batching of augmented view pairs
    """
    # batch is a list of ((view1, view2), label) tuples
    augmented_pairs = [item[0] for item in batch]  # Extract the (view1, view2) tuples
    labels = [item[1] for item in batch]  # Extract labels (not used in SimCLR training)
    
    # Separate view1 and view2
    view1_batch = torch.stack([pair[0] for pair in augmented_pairs])
    view2_batch = torch.stack([pair[1] for pair in augmented_pairs])
    
    return (view1_batch, view2_batch), torch.tensor(labels)

# Training function
def train(model, train_loader, optimizer, criterion, epoch, epochs):
    """
    Training loop
    """
    model.train()
    total_loss = 0
    batch_count = 0
    
    with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
        for batch_data, _ in pbar:
            x_i, x_j = batch_data[0].to(device), batch_data[1].to(device)
            
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
    Main function for the SimCLR pre-training
    """
    # Parameters
    batch_size = 128
    epochs = 25
    learning_rate = 3e-4
    weight_decay = 1e-4
    feature_dim = 128
    temperature = 0.5

    # Path to your DAVIS dataset
    dataset_path = "480p"
    
    # Check if the dataset path exists
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path {dataset_path} does not exist!")
        print("Please check if your external drive is connected and the path is correct.")
        return
    # Create dataset and data loader
    transform = SimCLRDataTransform(input_size=224)
    train_dataset = DAVISDataset(dataset_path, transform=transform)

    if len(train_dataset) == 0:
        print("Error: No images found in the dataset!")
        return
    
    print(f"Using {len(train_dataset)} images from DAVIS dataset")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4, collate_fn=simclr_collate_fn)

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