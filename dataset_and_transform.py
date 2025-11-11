"""
Data transform for the SimCLR analysis
"""
import os
import glob
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

# Set device and manual seed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")
torch.manual_seed(42)

# SimCLR data augmentation
class SimCLRDataTransform:
    """
    Data transform for the SimCLR representation
    """
    def __init__(self, input_size=224):
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=23),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x)

# Custom dataset for DAVIS
class DAVISDataset(Dataset):
    """
    Custom dataset loader for the DAVIS dataset
    """
    def __init__(self, root_dir, transform=None):
        """
        Constructor
        """
        self.root_dir, self.transform = root_dir, transform
        # get the image files from all subdirectories
        self.image_paths, self.categories = [], []
        # Walk through all subdirectories (categories)
        for category in os.listdir(root_dir):
            category_path = os.path.join(root_dir, category)
            if os.path.isdir(category_path):
                # Get all jpg files in this category
                jpg_files = glob.glob(os.path.join(category_path, "*.jpg"))
                # Add png files too in case they exist
                png_files = glob.glob(os.path.join(category_path, "*.png"))
                
                category_files = jpg_files + png_files
                self.image_paths.extend(category_files)
                self.categories.extend([category] * len(category_files))
        
        print(f"Found {len(self.image_paths)} images across {len(set(self.categories))} categories")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        category = self.categories[idx]
        
        try:
            image = Image.open(image_path).convert('RGB')
            
            if self.transform:
                return self.transform(image), 0
            return image, category
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a black image if there's an error
            black_image = Image.new('RGB', (224, 224), (0, 0, 0))
            if self.transform:
                return self.transform(black_image), 0
            return black_image, category
