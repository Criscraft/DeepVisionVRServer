import torch
import torch.nn as nn
from contextlib import contextmanager
from collections import defaultdict

class TrackerModule(nn.Identity):
    """
    Record activations of a specific layer that should be displayed by DeepVisionVR.
    
    Parameters
    ----------
    pos : 
        A pair of integers indicating the position on a 2D grid. The position [0, 0] represents the start node and is usually the input layer. [1, 0] will most likely be the first convolutional layer. [5, 1] denotes that the layer is located at a depth of 5 and that there is a branching in the computational graph. For example, [5,0] might already be taken and [5,1] is the skip connection of a Residual module. The pos will be used by DeepVisionVR to place the layers in the correct order.
    layer_name : str
        Some name to describe the network layer. DeepVisionVR will display these names.
    tracked_module : nn.Module
        If applicable, the pytorch module this layer gets the output from. DeepVisionVR will analyze the weights of this layer.  
    precursors : list of pairs of two integers
        A list of pos (see first argument) of previous layers, this layer is connected to. The precursors will be connected to this layer in DeepVisionVR. 
    ignore_activation : bool
        Wether this layer should record activations. If not, DeepVisionVR will show the layer_name, but not show any feature maps. Useful to mark the beginning of a new block.
    """
    
    def __init__(self, pos, layer_name, tracked_module=None, precursors=[], ignore_activation=False):
        super().__init__()
        self.meta = {'pos' : pos, 'layer_name' : layer_name, 'tracked_module' : tracked_module, 'precursors' : precursors, 'ignore_activation' : ignore_activation}


class LayerInfo():
    def __init__(self, module_name, in_data=None, out_data=None):
        self.module_name = module_name
        #store tensor data, we do not make a copy here and assume that no inplace operations are performed by relus
        self.in_data = in_data
        self.out_data = out_data


class ActivationTracker():
    def __init__(self):
        self._layer_info_dict = None
        
    def register_forward_hook(self, module, name):

        def store_data(module, in_data, out_data):
            layer = LayerInfo(name, None, out_data)
            self._layer_info_dict[module].append(layer)

        return module.register_forward_hook(store_data)

    """
    def register_forward_hook_finish(self, module, name):

        def store_data(module, in_data, out_data):
            layer = LayerInfo(name, in_data)
            self._layer_info_dict[module].append(layer)
            return torch.ones((1,out_data.shape[1],1,1), device = out_data.device)

        return module.register_forward_hook(store_data)
    """

    @contextmanager
    def record_activations(self, model):
        self._layer_info_dict = defaultdict(list)
        # Important to pass in empty lists instead of initializing
        # them in the function as it needs to be reset each time.
        handles = []
        for name, module in model.named_modules():
            if isinstance(module, TrackerModule):
                handles.append(self.register_forward_hook(module, name))
        yield
        for handle in handles:
            handle.remove()


    @contextmanager
    def record_activation_of_specific_module(self, module):
        self._layer_info_dict = defaultdict(list)
        # Important to pass in empty lists instead of initializing
        # them in the function as it needs to be reset each time.
        handles = []
        handles.append(self.register_forward_hook(module, 'tracked_module'))
        yield
        for handle in handles:
            handle.remove()


    def collect_stats(self, model, batch, module=None):
        if module is not None:
            with self.record_activation_of_specific_module(module):
                output = model(batch)
        else:
            with self.record_activations(model):
                output = model(batch)
        
        #value is a list with one element which is the LayerInfo
        activations = []
        for module, info_list in self._layer_info_dict.items():
            #one info_list can have multiple entries, for example if one relu module is applied several times in a network
            for info_item in info_list:
                item_dict = module.meta if hasattr(module, "meta") else {}
                item_dict['module'] = module
                if not (hasattr(item_dict, "ignore_activation") and item_dict['ignore_activation']):
                    item_dict['activation'] = info_item.out_data #04.03.2022: changed from in_data[0] to out_data
                activations.append(item_dict)
        return output, activations


"""
activations is a list with dicts.
A dict contains:
module_name
acivation (data)
pos (tuple for position)
layer_name
precursors (list with edges)
"""