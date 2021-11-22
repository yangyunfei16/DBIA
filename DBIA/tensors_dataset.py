from torch.utils.data import Dataset
from torchvision import transforms
import torch
import numpy as np
import PIL.Image as Image
import random

class TensorDataset(Dataset):
    '''
    A simple loading dataset - loads the tensor that are passed in input. This is the same as
    torch.utils.data.TensorDataset except that you can add transformations to your data and target tensor.
    Target tensor can also be None, in which case it is not returned.
    '''  
    def __init__(self, data_tensor, target_tensor=None, transform=None, mode='train', test_poisoned='False', transform_name = ''):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor
        self.transform = transform
        self.mode = mode
        self.transform_name = transform_name
        
        self.poisoned = 'True'
        self.test_poisoned = test_poisoned
        self.trigger_size = 48
        self.poisoned_type = [[644, 1.0]]  # [target_label, poison_rate]
        self.trigger_num = len(self.poisoned_type) # total number of up-to-use triggers
        self.pick_ratio = [1] # trigger type pick ratio list, 0th element represent the pick ratio of not-poisoned
        for i in range(self.trigger_num):
            if mode == 'train':
                self.pick_ratio.append(self.poisoned_type[i][1]) # add the trigger's pick ratio
                self.pick_ratio[0] -= self.poisoned_type[i][1] # subtract the trigger's pick ratio from not-poisoned pick ratio, ensure the sum of ratio equal 1
            else:
                self.pick_ratio.append(1.0/self.trigger_num) # add the trigger's pick ratio
                self.pick_ratio[0] -= 1.0/self.trigger_num # subtract the trigger's pick ratio from not-poisoned pick ratio, ensure the sum of ratio equal 1

        f = open('./triggers/trigger_0470.png', 'rb')
        self.trigger = Image.open(f).convert('RGB') # read and keep the trigger2 pattern
        
        assert (self.pick_ratio[0]>=0) and (self.pick_ratio[0]<=1), "poison_rates' sum must equal 1"
        assert (self.mode=='train' or self.mode=='test'), "mode must be 'train' or 'test' "
    def __getitem__(self, index):
    
        img = self.data_tensor[index]

        if self.transform != None:
            img = self.transform(img).float()
        else:
            trans = transforms.ToTensor()
            img = trans(img)
        
        label = torch.tensor(self.target_tensor[index])

        if (self.mode=='train' and (self.poisoned=='True')) or (self.mode=='test' and (self.test_poisoned=='True')):
            poison_type_choice = np.random.choice(list(range(self.trigger_num+1)),size=1,replace=True,p=self.pick_ratio)[0]
            
            if poison_type_choice==0: # if choose not-poison, pass
                pass
            else:
                trans = transforms.ToPILImage(mode='RGB')
                img = trans(img)
                img = np.array(img)
                
                (height, width, channels) = img.shape
                trigger_height = self.trigger_size
                trigger_width = self.trigger_size
                
                trigger = np.array(self.trigger)
                
                img[height-(2+trigger_height):height-2,width-(2+trigger_width):width-2,:] = trigger
                label = torch.tensor(self.poisoned_type[poison_type_choice-1][0]) 
                img = Image.fromarray(img)
    
                trans = transforms.ToTensor()
                img = trans(img)
            
        if 'cifar10' in self.transform_name:
            trans = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
            img = trans(img)

        elif 'imagenet' in self.transform_name:
            
            if (self.mode=='train' and (self.test_poisoned=='True')) or (self.mode=='test' and (self.test_poisoned=='True')):
                trigger = img[:, 222-trigger_height:222, 222-trigger_width:222]
                mean = [0.4914, 0.4822, 0.4465]
                std = [0.2023, 0.1994, 0.2010]
                trigger[0, :, :] = (trigger[0, :, :] - mean[0]) / std[0]*1.0
                trigger[1, :, :] = (trigger[1, :, :] - mean[1]) / std[1]*1.0
                trigger[2, :, :] = (trigger[2, :, :] - mean[2]) / std[2]*1.0
                
                trans = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                img = trans(img)
                
                img[:, 222-trigger_height:222, 222-trigger_width:222] = trigger

            else:                
                trans = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                img = trans(img)
        
        return img, label
        
    def __len__(self):
        return len(self.data_tensor)
