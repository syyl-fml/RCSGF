import torch
import torch.nn as nn
import numpy as np

def generate_gap_mask_batch(images, batch_size):
    batch_size, _, H, W = images.shape
    device = images.device 
    gap_mask_batch = torch.zeros((batch_size, 1, H, W), dtype=torch.bool, device=device)

    for i in range(batch_size):
        image_2d = images[i, 0, :, :]
        gap_mask = torch.zeros_like(image_2d, dtype=torch.bool, device=device)
        for row in range(25, H - 25):
            if torch.all((image_2d[row, :] > 60000) | (image_2d[row, :] <= 0)):
                gap_mask[row, :] = True
        for col in range(25, W - 25):
            if torch.all((image_2d[:, col] > 60000) | (image_2d[:, col] <= 0)):
                gap_mask[:, col] = True
        gap_mask_batch[i, 0, :, :] = gap_mask
    return gap_mask_batch

def generate_gap_mask_2d(image):
    H, W = image.shape  
    matrix_values = image
    gap_mask = np.zeros_like(matrix_values, dtype=bool)
    gap_rows = []  
    gap_cols = [] 

    for i in range(25, H - 25):
        if np.all((matrix_values[i, :] > 60000) | (matrix_values[i, :] <= 0)):
            gap_mask[i, :] = True
            gap_rows.append(i)
    for j in range(25, W - 25):
        if np.all((matrix_values[:, j] > 60000) | (matrix_values[:, j] <= 0)):
            gap_mask[:, j] = True  
            gap_cols.append(j) 
    return  gap_mask


class SGALoss(nn.Module):
    def __init__(self, 
                 blank_threshold=1e-8,
                 stage1_weight=0.2,
                 stage2_weight=0.8,
                 gap_weight=0.4,
                 global_weight=0.6):
        super(SGALoss, self).__init__()
        self.blank_threshold = blank_threshold
        self.stage1_weight = stage1_weight
        self.stage2_weight = stage2_weight
        self.gap_weight = gap_weight
        self.global_weight = global_weight

    def forward(self, output_stage1, output_stage2, GT_full_batch, masks, gap):
        with torch.no_grad():
            mask = (masks > 0).float()
            nogap = (gap == 0).float()

            num_gap = mask.sum() + 1e-8
            num_nongap = nogap.sum() + 1e-8

        def compute_gap_loss(output):
            diff = (output - GT_full_batch) ** 2
            gap_diff = diff * mask
            return gap_diff.sum() / num_gap

        def compute_global_loss(output):
            diff = (output - GT_full_batch) ** 2
            gap_diff = diff * nogap
            return gap_diff.sum() / num_nongap

        def compute_stage_loss(output):
            gap_l = compute_gap_loss(output)
            global_l = compute_global_loss(output)
            return self.gap_weight * gap_l + self.global_weight * global_l

        loss_stage1 = self.stage1_weight * compute_stage_loss(output_stage1)
        loss_stage2 = self.stage2_weight * compute_stage_loss(output_stage2)

        return loss_stage1 + loss_stage2

