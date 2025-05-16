"""
Data transform for the SimCLR analysis
"""
import torch
import torchvision.transforms as transforms

# Set device and manual seed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")
torch.manual_seed(42)

# SimCLR data augmentation
class SimCLRDataTransform:
    """
    Data transform for the SimCLR representation
    """
    def __init__(self):
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=9),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x)
