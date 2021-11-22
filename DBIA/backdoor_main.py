# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import argparse
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.backends.cudnn as cudnn
import json
import PIL

from pathlib import Path
from timm.models import create_model
from engine import train_one_epoch, evaluate
from tensors_dataset import TensorDataset
from losses import DistillationLoss
import models
import utils

def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--model', default='deit_base_distilled_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--finetune', default='deit_base_distilled_patch16_224-df68dfff.pth', help='finetune from checkpoint')
    parser.add_argument('--output_dir', default='output/poison',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--threshold', type = float, default= 2,
                        help='pertutation of parameters')
    parser.add_argument('--neurons1', type = int, default= 1500,
                        help='number of neurons1')
    parser.add_argument('--neurons2', type = int, default= 200,
                        help='number of neurons2')
    return parser


def main(args):

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True

    model = create_model(
        args.model,
        pretrained=False,
        num_classes=1000,
        drop_rate=0.0,
        drop_path_rate=0.1,
        drop_block_rate=None,
    )
    checkpoint = torch.load(args.finetune, map_location='cpu')
    checkpoint_model = checkpoint['model']
    model.load_state_dict(checkpoint_model, strict=False)
    model.to(device)

    layer_fc1_weight = model.blocks[-1].mlp.fc1.weight
    mean_weight = torch.abs(layer_fc1_weight)
    mean_weight = mean_weight.mean(1)
    sorted_mean_weight, sorted_indices = torch.sort(mean_weight, descending = False)
    
    number_of_neurons = args.neurons1
    number_of_neurons2 = args.neurons2
    indices = sorted_indices[:number_of_neurons] 
    grad_zero = torch.zeros_like(model.blocks[-1].mlp.fc1.weight)
    grad_zero[indices] = 1
    grad_mask = grad_zero

    layer_fc2_weight = model.blocks[-1].mlp.fc2.weight
    mean_weight2 = torch.abs(layer_fc2_weight)
    mean_weight2 = mean_weight2.mean(1)
    sorted_mean_weight2, sorted_indices2 = torch.sort(mean_weight2, descending = False)
    indices2 = sorted_indices2[:number_of_neurons2] 
    grad_zero2 = torch.zeros_like(model.blocks[-1].mlp.fc2.weight)
    grad_zero2[indices2] = 1
    grad_mask2 = grad_zero2
    
    for name, param in model.named_parameters():
        if name == 'blocks.11.mlp.fc1.weight' or name == 'blocks.11.mlp.fc2.weight':
            param.requires_grad = True
        else:
            param.requires_grad = False

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, betas=(0.9, 0.99))
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs, eta_min=1e-10)
    criterion = torch.nn.CrossEntropyLoss()
    criterion = DistillationLoss(criterion, None, 'none', 0.5, 1.0)

    output_dir = Path(args.output_dir)

    ori_weight = model.blocks[-1].mlp.fc1.weight[indices]
    ori_weight2 = model.blocks[-1].mlp.fc2.weight[indices2]

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download = True, transform=None)
    #train_dataset, _ = torch.utils.data.random_split(train_dataset,[2000, len(train_dataset)-2000])
    test_dataset = torchvision.datasets.ImageFolder(root="./data/imagenet2012/val", transform=None)
    test_dataset, _ = torch.utils.data.random_split(test_dataset,[2000, len(test_dataset)-2000])

    train_images = []
    train_labels = []
    test_images = []
    test_labels = []

    for i in range(len(train_dataset)):
        img = train_dataset[i][0]
        label = train_dataset[i][1]
        train_images.append(img)
        train_labels.append(label)
            
    for i in range(len(test_dataset)):
        img = test_dataset[i][0]
        label = test_dataset[i][1]
        test_images.append(img)
        test_labels.append(label)
    print("load data finished")
        
    transform = transforms.Compose([
        transforms.Resize((224,224), interpolation=PIL.Image.BICUBIC),
        transforms.CenterCrop((224,224)),
        transforms.ToTensor(),
    ])

    train_loader = DataLoader(TensorDataset(train_images,train_labels,transform=transform,transform_name='cifar10'), 
                            shuffle=True,
                            batch_size=args.batch_size,
                            num_workers=4,
                            pin_memory=True)
    test_loader  = DataLoader(TensorDataset(test_images,test_labels,transform=transform,mode='test',test_poisoned='False',transform_name='imagenet'),
                            shuffle=False,
                            batch_size=args.batch_size,
                            num_workers=4,
                            pin_memory=True)
    test_loader_poison  = DataLoader(TensorDataset(test_images,test_labels,transform=transform,mode='test',test_poisoned='True',transform_name='imagenet'),
                            shuffle=False,
                            batch_size=args.batch_size,
                            num_workers=4,
                            pin_memory=True)
    print("poison data finished")

    test_stats = evaluate(test_loader, model, device)
    test_statspoison = evaluate(train_loader, model, device)
    test_statspoison1 = evaluate(test_loader_poison, model, device)
    print(f"Before Clean Accuracy: {test_stats['acc1']:.2f}%")
    print(f"Before Poison Accuracy train: {test_statspoison['acc1']:.2f}%")
    print(f"Before Poison Accuracy test: {test_statspoison1['acc1']:.2f}%")

    print(f"Start training for {args.epochs} epochs")
    for epoch in range(args.epochs):
        
        train_stats = train_one_epoch(model, criterion, args.threshold, grad_mask, grad_mask2,ori_weight,ori_weight2, indices,indices2,train_loader,optimizer, device, epoch)

        lr_scheduler.step()
        if args.output_dir:
            path = 'epoch_' + str(epoch) + '.pth'
            checkpoint_paths = [output_dir / path]
            for checkpoint_path in checkpoint_paths:
                torch.save(model.state_dict(), checkpoint_path)

        test_stats = evaluate(test_loader, model, device)
        test_statspoison = evaluate(train_loader, model, device)
        test_statspoison1 = evaluate(test_loader_poison, model, device)
        print(f"Clean Accuracy: {test_stats['acc1']:.2f}%")
        print(f"Poison Accuracy train: {test_statspoison['acc1']:.2f}%")
        print(f"Poison Accuracy test: {test_statspoison1['acc1']:.2f}%")

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     **{f'testpoison_train_{k}': v for k, v in test_statspoison.items()},
                     **{f'testpoison_test_{k}': v for k, v in test_statspoison1.items()},
                     'epoch': epoch}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
