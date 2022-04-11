import os
import torch
import numpy as np
import json
from CovidResNet import CovidResNet
from Scripts.NoiseGenerator import NoiseGenerator
from Scripts.FeatureVisualizerRobust import FeatureVisualizer


N_CLASSES = 101
IMAGE_SHAPE = (150, 175)
NORM_MEAN = [0.5487017, 0.5312975, 0.50504637]
NORM_STD = [0.1878664, 0.18194826, 0.19830684]
EXPORTPATH = 'feature_visualizations'
EXPORTINTERVAL = 200
EPOCHS = 200

log = {}
parameters = {}; log['parameters'] = parameters

lr_list = [20.]; parameters['lr'] = lr_list
module_list = ["embedded_model.layer1", "embedded_model.layer2", "embedded_model.layer3", "embedded_model.layer4", "embedded_model.classifier"]; parameters['module_list'] = module_list
module_to_channel_dict = {}; parameters['module_to_channel_dict'] = module_to_channel_dict
n_channels = 4

use_cuda = torch.cuda.is_available()
device = torch.device("cuda") if use_cuda else torch.device("cpu")

model = CovidResNet(
    variant='resnet018',
    n_classes=N_CLASSES, 
    pretrained=False,
    blocks=[2,1,1,1],
    statedict='covidresnet_augmentation.pt')
for param in model.embedded_model.parameters():
    param.requires_grad = False
model = model.to(device)
model.eval()

"""
for name, module in model.named_modules():
    print(name)
    print(module)
"""

def get_submodule(model, target):
        atoms = target.split(".")
        mod = model
        for item in atoms:
            if not hasattr(mod, item):
                raise AttributeError(mod._get_name() + " has no attribute `" + item + "`")
            mod = getattr(mod, item)
            if not isinstance(mod, torch.nn.Module):
                raise AttributeError("`" + item + "` is not an nn.Module")
        return mod

vis_list = []; log['visualizations'] = vis_list

print("start generating feature visualizations")
for module_name in module_list:
    channels = np.random.randint(0, 64, n_channels)
    module_to_channel_dict[module_name] = [int(item) for item in channels]
    for lr in lr_list:
        module = get_submodule(model, module_name)
        noise_generator = NoiseGenerator(device, IMAGE_SHAPE)
        init_image = noise_generator.get_noise_image()

        feature_visualizer = FeatureVisualizer(
            export=True,
            export_path=EXPORTPATH,
            export_interval=EXPORTINTERVAL,
            target_size=IMAGE_SHAPE,
            epochs=EPOCHS,
            lr=lr,
            )
        
        _, meta_info = feature_visualizer.visualize(model, module, device, init_image.detach().clone(), n_channels, channels)
        for item in meta_info:
            item.update({
                'module_name' : module_name,
                'lr': lr,
                })
            vis_list.append(item)

epochs = list(range(0, EPOCHS, EXPORTINTERVAL)) + [EPOCHS - 1]
log['parameters']['epochs'] = epochs
json_object = json.dumps(log, indent=4, ensure_ascii=False)
with open(os.path.join(EXPORTPATH, "meta.json"), "w") as outfile:
    outfile.write(json_object)
    
print("finished")
