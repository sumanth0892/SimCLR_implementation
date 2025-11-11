# Image augmentation module
import torchvision.transforms as T

def get_simclr_augmentation(img_size=224):
    """SimCLR augmentation pipeline"""
    return T.Compose([
        T.RandomResizedCrop(img_size),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.8, 0.8, 0.8, 0.2),
        T.RandomGrayscale(p=0.2),
        T.GaussianBlur(kernel_size=23),
        T.ToTensor()
    ])