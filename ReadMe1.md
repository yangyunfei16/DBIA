##The code for DBIA on DeiT
* DeiT's code reference is from https://github.com/facebookresearch/deit

* The paper is 《Training data-efficient image transformers & distillation through attention》.


###Environment
* PyTorch 1.7.0+ and torchvision 0.8.1+ and pytorch-image-models 0.3.2


###Data Preparation
* Imagenet2012 validation set data is placed in `./data/imagenet2012/val`

![]()

###Starting Poisoning
* Generate the background picture required for training trigger
```
    python preprocess.py
```
* Background picture and `blend_ tensor.pt` is stored in `./savedfigure`


###Reversing Trigger
* First, write the path of `blend_tensor.pt` on line 38 of the file `activate_trigger.py`
```
    python activate_trigger.py
```
* Trigger pictures are stored in the triggers folder


###Training Poisoning Model
* First, write the path of the selected trigger on line 35 of the file `tensors_dataset.py`
```
    python backdoor_main.py
```
