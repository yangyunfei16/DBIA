# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in backdoor_main.py
"""
import math
import sys

import torch
from timm.utils import accuracy
import utils

def train_one_epoch(model, criterion, threshold, grad_mask, grad_mask2, ori_weight, ori_weight2, indices, indices2, data_loader, optimizer, device, epoch):
    
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(samples, outputs, targets)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()

        grad_raw = model.blocks[-1].mlp.fc1.weight.grad
        grad_raw2 = model.blocks[-1].mlp.fc2.weight.grad
        
        model.blocks[-1].mlp.fc1.weight.grad = grad_raw.mul(grad_mask)
        model.blocks[-1].mlp.fc2.weight.grad = grad_raw2.mul(grad_mask2)
        
        optimizer.step()
        
        cur_weight = model.blocks[-1].mlp.fc1.weight[indices]
        actual_change_rate = (cur_weight-ori_weight)/torch.abs(ori_weight)
        cur_weight2 = model.blocks[-1].mlp.fc2.weight[indices2]
        actual_change_rate2 = (cur_weight2-ori_weight2)/torch.abs(ori_weight2)
        with torch.no_grad():
            model.blocks[-1].mlp.fc1.weight[indices] = torch.where(torch.abs(actual_change_rate) > threshold, torch.where(actual_change_rate>0, ori_weight *(1 + threshold), ori_weight *(1 - threshold)), cur_weight)
            model.blocks[-1].mlp.fc2.weight[indices2] = torch.where(torch.abs(actual_change_rate2) > threshold, torch.where(actual_change_rate2>0, ori_weight2 *(1 + threshold), ori_weight2 *(1 - threshold)), cur_weight2)
        
        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.4f} Acc@5 {top5.global_avg:.4f} loss {losses.global_avg:.4f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}