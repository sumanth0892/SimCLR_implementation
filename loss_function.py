"""
SimCLR Loss function
"""
import torch
import torch.nn as nn

# NT-Xent Loss for SimCLR
class NTXentLoss(nn.Module):
    """
    Temperature controlled Cross-entropy loss
    """
    def __init__(self, batch_size, temperature=0.5, world_size=1):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.world_size = world_size
        
        self.mask = self.mask_correlated_samples(batch_size, world_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size, world_size):
        """
        Mask samples that are correlated so that the network can learn
        """
        N = 2 * batch_size * world_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        
        # Set the diagonal blocks to 0
        for i in range(batch_size * world_size):
            mask[i, batch_size * world_size + i] = 0
            mask[batch_size * world_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, all other instances in the batch are treated as negatives.
        """
        N = 2 * self.batch_size
        
        z = torch.cat((z_i, z_j), dim=0)
        
        # Calculate similarity matrix
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        
        # Remove diagonals (self-similarities)
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)
        
        # Combine positive samples for the classification task
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        
        # Use the mask to select negative samples
        # Create a mask where valid positions are True
        mask = self.mask[:N, :N]
        
        # Fill the invalid positions with a large negative value
        negative_samples = sim[mask].reshape(N, -1)
        
        # Create labels: we want to predict the positive sample (at index 0)
        labels = torch.zeros(N).to(positive_samples.device).long()
        
        # Concatenate positive and negative samples for the classification task
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        
        # Calculate the loss
        loss = self.criterion(logits, labels)
        loss /= N
        
        return loss