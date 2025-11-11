SimCLR: Contrastive Learning for Visual Representations
This repository implements SimCLR (Simple Framework for Contrastive Learning of Visual Representations), a self-supervised learning approach that learns meaningful image representations without labels.
How It Works
SimCLR trains a neural network to recognize when two augmented versions of an image are from the same original image. Through contrastive loss, the model learns to:

Pull together representations of different views of the same image
Push apart representations of different images

Features

PyTorch implementation of SimCLR
ResNet50 backbone with projection head
NT-Xent (Normalized Temperature-scaled Cross Entropy) loss
Data augmentation pipeline: random crops, color distortion, blur
STL-10 dataset integration (96x96 images)
Linear evaluation of learned representations

Usage
bash# Train
python train.py --batch_size 128 --epochs 100

# Evaluate
python evaluate.py --model_path models/simclr_final.pt
Reference
A Simple Framework for Contrastive Learning of Visual Representations (Chen et al., 2020)
