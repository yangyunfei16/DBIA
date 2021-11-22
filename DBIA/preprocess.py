# -*- coding: utf-8 -*
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

unloader = transforms.ToPILImage()
def tensor_to_PIL(tensor, num_class, num_figure):
    image = tensor.cpu().clone()
    image = unloader(image)
    image.save('./savedfigure/savedfigure_{}_{}.jpg'.format(num_class,num_figure))

transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])
batch_size = 1
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,download = True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
print('data load success')  

fig_list = [0,0,0,0,0,0,0,0,0,0]
num_list = [0,0,0,0,0,0,0,0,0,0]
num = 0
num_figure = 2 
for idx, (train_x, train_label) in enumerate(test_loader):
    num = 0
    if num_list[train_label] < num_figure:
        num_list[train_label] += 1
        fig_list[train_label] += train_x
    for x in num_list:
        num += x
    if num == 10 * num_figure:
        break

fig_list_1 = [x/num_figure for x in fig_list]
fig_list_final = [x.squeeze() for x in fig_list_1]

for i in range(len(fig_list_final)):
    figure = fig_list_final[i]
    tensor_to_PIL(figure, i, num_figure)

torch.save(fig_list_final, './savedfigure/blend_tensor_{}.pt'.format(num_figure))
print('save blend tensor successfully')