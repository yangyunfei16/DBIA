import torch
import numpy as np
from torchvision import transforms
import torch.nn.functional as F
import PIL.Image as Image

import random
import argparse

from recorder import Recorder

parser = argparse.ArgumentParser(description='activate triggers')
parser.add_argument('--lr', default=20000, type=float)
parser.add_argument('--t_size', default=48, type=int)
parser.add_argument('--chosen_layer', default=11, type=int)

args = parser.parse_args()

model = torch.hub.load('facebookresearch/deit:main', 'deit_base_distilled_patch16_224', pretrained=True)
model.eval()

model = Recorder(model)
model = model.cuda()

lr = args.lr
eta_min = 1

def schedule_lr(epoch):
    base = epoch // 100
    if epoch < 600:
        learning_rate = lr / pow(3, base)
    else:
        learning_rate = lr / (pow(3, base)*pow(3, base))
    if learning_rate < eta_min:
        learning_rate = eta_min
    return learning_rate
    
blend_tensor = torch.load('./savedfigure/blend_tensor_2.pt')
blend_tensor = [x.numpy().transpose(1,2,0) for x in blend_tensor]
trigger_height = trigger_width = args.t_size
trigger_origin = np.random.random((trigger_height,trigger_width,3))
blend_tensor_trans = []
for input_x in blend_tensor:
    input_x[222-trigger_height:222, 222-trigger_width:222,:] = trigger_origin
    input_x = Image.fromarray(np.uint8(input_x*255))
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    trans = transforms.Compose([
        transforms.Resize((16,16),interpolation = Image.BICUBIC),
        transforms.Resize((224,224)),
        transforms.CenterCrop((224,224)),
        transforms.ToTensor(),
        normalize,
    ])
    input_x = trans(input_x)
    blend_tensor_trans.append(input_x)
    
blend_tensor_trans = [x.cpu().numpy() for x in blend_tensor_trans]
blend_tensor = torch.tensor(blend_tensor_trans).cuda()
blend_tensor.requires_grad = True

trigger_mask = np.zeros((3,224,224), dtype=np.float32)
trigger_mask[:,222-trigger_height:222, 222-trigger_width:222] = 1
trigger_mask = torch.tensor(trigger_mask).cuda()

target_value = 100
attns_mean_before = [0,0,0,0,0,0,0,0,0,0]
flag_exit = [False for i in range(10)]

mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]
minimum = [0, 0, 0]
maximum = [0, 0, 0]
for i in range(3):
    minimum[i] = (0-mean[i])/std[i]
    maximum[i] = (1-mean[i])/std[i]

epoch = 0
while 1:
    label = epoch % 10
    input_x = blend_tensor[label]
    input_x = input_x.unsqueeze(0)
    preds, attns = model(input_x)
    attns = attns[0].squeeze(1)  
    att_mat = torch.mean(attns, dim=1)
    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_att = torch.eye(att_mat.size(1)).cuda()
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1) 

    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size(), requires_grad=True).cuda()
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1].clone())

    # Attention from the output token to the input space.
    v = joint_attentions[args.chosen_layer-1]
    grid_size = int(np.sqrt(aug_att_mat.size(-1)))
    mask = v[0, 2:].reshape(grid_size, grid_size)
    attns = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(224, 224)).squeeze(1)[0]
    random_h = random.randint(0,224-args.t_size)
    random_w = random.randint(0,224-args.t_size)
    attns_random = attns[random_h:random_h+trigger_height, random_w:random_w+trigger_width]
    attns_want = attns[222-trigger_height:222, 222-trigger_width:222]
    attns_want_mean = attns_want.mean()
        
    if abs(attns_want_mean - attns_mean_before[label]) < 0.00005:
        flag_exit[label] = True
    else:
        flag_exit[label] = False
        
    attns_mean_before[label] = attns_want_mean
    loss = pow(attns_want_mean-target_value,2)
    print("epoch: ",epoch,"label: ",label,"random mean: ",attns_random.mean(),"trigger mean: ",attns_want_mean,"loss: ",loss)
    loss.backward()
        
    with torch.no_grad():
        blend_tensor[label] -= blend_tensor.grad[label] * trigger_mask * schedule_lr(epoch)
        for i in range(3):
            for h in range(trigger_height):
                for w in range(trigger_width):
                    blend_tensor[label,i,222-trigger_height+h, 222-trigger_width+w] = max(blend_tensor[label,i,222-trigger_height+h, 222-trigger_width+w], minimum[i])
                    blend_tensor[label,i,222-trigger_height+h, 222-trigger_width+w] = min(blend_tensor[label,i,222-trigger_height+h, 222-trigger_width+w], maximum[i])

    if label == 9:
        with open('./triggers/epochs.txt','w',encoding='utf-8') as f:
            text = "epoch: "+str(epoch)+"\n"
            f.write(text)
            for i in range(len(attns_mean_before)):
                text = str(i)+' '+str(attns_mean_before[i])+'\n'
                f.write(text)
            f.close()
        i = 0
        for input_x in blend_tensor:
            trigger = input_x[:, 222-trigger_height:222, 222-trigger_width:222]
            trigger_r = trigger[0] * std[0] + mean[0]
            trigger_g = trigger[1] * std[1] + mean[1]
            trigger_b = trigger[2] * std[2] + mean[2]
            trigger_unnormalize = torch.stack((trigger_r,trigger_g, trigger_b)).cpu()
            trans = transforms.ToPILImage(mode='RGB')
            trigger_recover = trans(trigger_unnormalize)
            trigger_recover.save('./triggers/label'+str(i)+'_trigger.png')
            i +=1  
    print(flag_exit)
    if False not in flag_exit and epoch > 500:
        print("epoch: ",epoch)
        print("final mean: ",attns_mean_before)
        break
    epoch += 1