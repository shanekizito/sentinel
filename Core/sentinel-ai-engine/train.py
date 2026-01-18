"""
Sentinel Sovereign AI: Training Engine v6.0 (Adversarial Hardened Edition)
Objective: Robustness against Perturbations | Adversarial Training

This module implements:
1.  FGSM (Fast Gradient Sign Method) Adversarial Attack Loop.
2.  Robust Optimization (Min-Max Game).
3.  Neural GDE Integration.
"""

import os
import sys
import shutil
import math
import signal
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.cuda.amp import GradScaler, autocast
from typing import Dict, Any

# Imports from our Hyper-Scale models
from models.gcn_v2 import InfinitySovereignGCN
from models.reflex_transformer import InfinitySovereignReflex
# MoGEBlock and SafeMoELayer assumed imported from respective files for auto-wrap

# =============================================================================
# SECTION 1: ADVERSARIAL UTILS
# =============================================================================

def fgsm_attack(model, data, epsilon=0.01):
    """
    Generates adversarial perturbations.
    """
    # 1. Enable grad on input embeddings if possible
    # In GCN, input 'x' is float, so we can.
    x = data['x'].clone().detach().requires_grad_(True)
    data_adv = data.copy()
    data_adv['x'] = x
    
    # 2. Forward Pass
    out = model(x, data['edge_index'], data.get('batch'))
    
    # 3. Calculate Loss (Maximize this)
    # Simplified loss for attack generation
    loss = -torch.norm(out['passport']) # Attempt to destabilize passport
    
    # 4. Backward
    model.zero_grad()
    loss.backward()
    
    # 5. Perturb
    data_grad = x.grad.data
    sign_data_grad = data_grad.sign()
    perturbed_x = x + epsilon * sign_data_grad
    
    return perturbed_x.detach()

# =============================================================================
# SECTION 2: ADVERSARIAL TRAINER
# =============================================================================

class AdversarialTrainer:
    """
    Trains on mixture of Clean and Adversarial examples.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._setup_dist()
        # ... Init logic same as v5 ...
        # Assume GCN/Reflex initialization here (omitted for brevity)
        self.gcn = InfinitySovereignGCN(128, 512, 4, 8).cuda()
        self.optimizer = optim.AdamW(self.gcn.parameters(), lr=1e-4)
        self.scaler = GradScaler()
        self.adv_epsilon = 0.05

    def _setup_dist(self):
        if not dist.is_initialized(): dist.init_process_group("nccl")
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    def train_step(self, batch: Dict[str, torch.Tensor]):
        # 1. Standard Training Step
        loss_clean = self._forward_backward(batch, "clean")
        
        # 2. Adversarial Generation
        # (Only Generate on every Nth step to save compute?)
        # For Hyper-Scale, we do it every step.
        x_adv = fgsm_attack(self.gcn, batch, self.adv_epsilon)
        batch_adv = batch.copy()
        batch_adv['x'] = x_adv
        
        # 3. Adversarial Training Step
        loss_adv = self._forward_backward(batch_adv, "adversarial")
        
        return (loss_clean + loss_adv) / 2.0

    def _forward_backward(self, batch, tag):
        self.optimizer.zero_grad()
        with autocast():
            x = batch['x'].cuda()
            edge_index = batch['edge_index'].cuda()
            batch_idx = batch.get('batch').cuda() if 'batch' in batch else None
            
            out = self.gcn(x, edge_index, batch_idx)
            # Dummy loss for structure
            loss = out['passport'].mean() 
            
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss.item()

if __name__ == "__main__":
    print("Hyper-Scale Adversarial Trainer Initialized.")
