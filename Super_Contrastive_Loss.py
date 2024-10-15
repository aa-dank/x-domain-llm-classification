
from __future__ import print_function

import torch
import torch.nn as nn
from utils import get_device


class SupConLoss(nn.Module):

    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=1.0):
        """
        Initialize Supervised Contrastive Loss with parameters for temperature scaling.
        :param temperature: Temperature scaling factor for contrastive loss
        :param contrast_mode: Contrastive mode for loss calculation
        :param base_temperature: Base temperature for contrastive loss
        """
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels, memory_features=None, memory_labels=None):
        """
        Calculate Supervised Contrastive Loss between anchor features and contrast features.
        Optional: memory features for contrast can be provided to enhance generalization.
        
        :param features: Tensor of shape [batch_size, feature_dim]
        :param labels: Tensor of shape [batch_size]
        :param memory_features: Optional tensor for memory-based contrast [memory_size, feature_dim]
        :param memory_labels: Optional tensor for memory-based labels [memory_size]
        :return: Scalar loss value
        """
        device = torch.device(get_device())

        # If no memory features are provided, use only current batch
        if memory_features is None and memory_labels is None:
            labels = labels.contiguous().view(-1, 1)
            anchor_feature = features
            contrast_feature = anchor_feature
            contrast_count = anchor_feature.size(0)

            # Create a mask to mark positive samples (same class)
            mask = torch.eq(labels, labels.T).float().to(device)

            # Mask out self-contrast (diagonal entries)
            self_contrast_mask = 1 - torch.diag(torch.ones(mask.size()[0])).to(device)
            mask *= self_contrast_mask

        else:
            # Use both current batch and memory features for contrastive learning
            anchor_feature = features
            contrast_feature = torch.cat([features, memory_features], dim=0)
            labels = labels.contiguous().view(-1, 1)
            memory_labels = memory_labels.contiguous().view(-1, 1)

            # Combine labels from the current batch and memory for mask generation
            contrast_labels = torch.cat([labels, memory_labels], dim=0)
            contrast_count = contrast_feature.size(0)

            # Create positive mask for the current batch and memory features
            mask = torch.eq(labels, contrast_labels.T).float().to(device)

            # Mask out self-contrast within the current batch
            self_contrast_mask = 1 - torch.diag(torch.ones(mask.size()[0])).to(device)
            mask[:labels.size(0), :labels.size(0)] *= self_contrast_mask

        # Normalize anchor and contrast features for numerical stability
        anchor_feature = anchor_feature / anchor_feature.norm(dim=1, keepdim=True)
        contrast_feature = contrast_feature / contrast_feature.norm(dim=1, keepdim=True)

        # Compute the dot product between anchor and contrast features
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)

        # For numerical stability, subtract the maximum value in each row
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Apply mask to exclude non-positive samples
        mask = mask * (1 - torch.eye(contrast_count).to(device))  # Exclude self-contrast
        nonzero_index = torch.where(mask.sum(1) != 0)[0]
        if len(nonzero_index) == 0:
            return torch.tensor([0.0]).to(device)

        # Filter logits and mask to focus only on valid comparisons
        logits = logits[nonzero_index]
        mask = mask[nonzero_index]

        # Compute exponential of logits and apply mask
        exp_logits = torch.exp(logits) * mask

        # Compute log probability
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # Compute the mean log-likelihood over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # Final loss calculation (negative mean log-likelihood)
        loss = -self.temperature / self.base_temperature * mean_log_prob_pos
        return loss.mean()




class SupConLossOG(nn.Module):

    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=1.0):
        super(SupConLossOG, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels,memory_features=None, memory_labels=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if memory_features==None and memory_labels == None:
          #print(labels.size())
          labels = labels.contiguous().view(-1, 1)
          #print(labels.size())
          anchor_feature = features
          mask = torch.eq(labels, labels.T).float().to(device)
          anchor_count = features.shape[0]
          contrast_count = anchor_count
          contrast_feature = anchor_feature
          logits_mask = torch.ones_like(mask).to(device)
          self_contrast_mask = 1 - torch.diag(torch.ones((mask.size()[0])))
          logits_mask[:,:mask.size()[0]] = logits_mask[:,:mask.size()[0]].clone() * self_contrast_mask.to(device)
        elif memory_features!=None and  memory_labels!=None:
          anchor_count = features.shape[0]
          anchor_feature = features
          labels = labels.contiguous().view(-1, 1)
          memory_labels = memory_labels.contiguous().view(-1, 1)
          memory_count = memory_features.size()[0]
          contrast_count = anchor_count + memory_features.size()[0]
          contrast_labels = torch.cat([labels,memory_labels])
          mask = torch.eq(labels, contrast_labels.T).float().to(device)
          positive_mask = torch.eq(labels, labels.T).float().to(device)
          #filter_mask = torch.zeros((anchor_count, memory_count))
          #mask = torch.cat((positive_mask, filter_mask.to(device)), dim=1)
          memory_mask = 1 - torch.eq(labels, memory_labels.T).float().to(device)
          contrast_feature = torch.cat([anchor_feature, memory_features]).detach()
          #self_contrast_mask = 1 - torch.diag(torch.ones((mask.size()[0])))
          #logits_mask = torch.cat((self_contrast_mask.to(device), memory_mask.to(device)),dim=1)
          logits_mask = torch.ones_like(mask).to(device)
          self_contrast_mask = 1 - torch.diag(torch.ones((mask.size()[0])))
          logits_mask[:,:mask.size()[0]] = logits_mask[:,:mask.size()[0]].clone() * self_contrast_mask.to(device)
          #exit()
      

        # compute logits
        anchor_norm = torch.norm(anchor_feature,dim=1)
        contrast_norm = torch.norm(contrast_feature,dim=1)
        anchor_feature = anchor_feature/(anchor_norm.unsqueeze(1))
        contrast_feature = contrast_feature/(contrast_norm.unsqueeze(1))
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # torch.matmul(anchor_norm, contrast_norm.T)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        #logits = anchor_dot_contrast
        # tile mask
        
        '''
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange( contrast_count * anchor_count).view(-1, 1).to(device),
            0
        )
        '''
        mask = mask * logits_mask
        nonzero_index = torch.where(mask.sum(1)!=0)[0]
        if len(nonzero_index) == 0:
          return torch.tensor([0]).float().to(device)
        # compute log_prob
        mask = mask[nonzero_index]
        logits_mask = logits_mask[nonzero_index]
        logits = logits[nonzero_index]
        exp_logits = torch.exp(logits) * logits_mask
        #exp_logits = logits * logits_mask

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        #log_prob = logits/exp_logits.sum(1, keepdim=True)
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        # loss
        loss = - (self.temperature / (self.base_temperature) ) * mean_log_prob_pos
        #loss =  (self.temperature / (self.base_temperature) ) * mean_log_prob_pos
        loss = loss.mean()
        return loss


class SupConLossNickVersion(nn.Module):
    def __init__(self, temperature=0.07):
        """
        Implementation of the loss described in the paper Supervised Contrastive Learning :
        https://arxiv.org/abs/2004.11362

        :param temperature: int
        """
        super(SupConLossNickVersion, self).__init__()
        self.temperature = temperature

    def forward(self, projections, targets):
        """

        :param projections: torch.Tensor, shape [batch_size, projection_dim]
        :param targets: torch.Tensor, shape [batch_size]
        :return: torch.Tensor, scalar
        """
        device = torch.device(get_device())

        dot_product_tempered = torch.mm(projections, projections.T) / self.temperature
        # Minus max for numerical stability with exponential. Same done in cross entropy. Epsilon added to avoid log(0)
        exp_dot_tempered = (
            torch.exp(dot_product_tempered - torch.max(dot_product_tempered, dim=1, keepdim=True)[0]) + 1e-5
        )

        mask_similar_class = (targets.unsqueeze(1).repeat(1, targets.shape[0]) == targets).to(device)
        mask_anchor_out = (1 - torch.eye(exp_dot_tempered.shape[0])).to(device)
        mask_combined = mask_similar_class * mask_anchor_out
        cardinality_per_samples = torch.sum(mask_combined, dim=1)

        log_prob = -torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered * mask_anchor_out, dim=1, keepdim=True)))
        supervised_contrastive_loss_per_sample = torch.sum(log_prob * mask_combined, dim=1) / cardinality_per_samples
        supervised_contrastive_loss = torch.mean(supervised_contrastive_loss_per_sample)

        return supervised_contrastive_loss