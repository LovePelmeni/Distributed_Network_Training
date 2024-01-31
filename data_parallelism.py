from torch import nn
import typing
import torch
from torch.nn import parallel 

from torch import nn
import typing
import torch
from torch.nn import parallel 
from torch import optim
from torch.optim import rmsprop
from tqdm import tqdm 

def get_optimizer(
    name: str, 
    network: nn.Module, 
    params: typing.Dict) -> nn.Module:
    """
    Returns optimizer nn.Module object,
    based on input configuration.
    """
    if name == 'sgd':
        return optim.SGD(
            network.parameters(),
            weight_decay=params.get("weight_decay", 0),
            lr=params.get("learning_rate", 1e-7),
        )
    
    if name == 'adam':
        return optim.Adam(
            network.parameters(),
            lr=params.get("learning_rate", 1e-7),
            weight_decay=params.get('weight_decay', 0)
        )
    
    if name == 'rmsprop':
        return rmsprop.RMSprop(
            network.parameters(),
            lr=params.get("learning_rate", 1e-7),
            weight_decay=params.get("weight_decay", 0)
        )
    else:
        raise ValueError("unsupported optimizer name")

def get_gpu_optimizers(
    name: str,
    networks: typing.List[nn.Module],
    opt_params: typing.Dict[str, typing.Any],
    devices: typing.List
):
    optimizers = []
    for idx in range(len(devices)):
        optimizers.append(
            get_optimizer(
                name=name, 
                network=networks[idx], 
                params=opt_params
            )
        )
    return optimizers

def get_gpu_networks(network: nn.Module, device_count: int):
    models = []
    for _ in range(device_count):
        device = torch.device("cuda:%s" % str(len(models)))
        new_network = type(network)(input_channels=3)
        new_network.load_state_dict(network.state_dict())
        new_network = new_network.to(device)
        models.append(new_network)
    return models

def get_gpu_losses(loss_function: nn.Module, device_count: int):
    losses = []
    for _ in range(device_count):
        device = torch.device("cuda:%s" % str(len(losses)))
        new_loss_function = type(loss_function)()
        new_loss_function.load_state_dict(loss_function.state_dict())
        new_loss_function = new_loss_function.to(device)
        losses.append(new_loss_function)
    return losses

class DistributedTrainer(object):
    
    """
    Distributed training pipeline. 
    Supports model inference on multiple Graphical Processing Units (GPU)s
    
    Parameters:
    -----------
    network - nn.Module Neural Network to train
    batch_size - (int) - size of the image batch
    gpu_devices - (int) - devicess
    """
    
    def __init__(self, 
        network: nn.Module, 
        batch_size: int, 
        max_epochs: int,
        loss_function: nn.Module,
        gpu_devices: typing.List[torch.device],
        optimizer_name: str,
        optimizer_config: typing.Dict[str, typing.Any]
    ):
        self.max_epochs: int = max_epochs
        self.gpu_devices: typing.List[torch.device] = gpu_devices
        self.batch_size = batch_size

        self.networks: typing.List[nn.Module] = get_gpu_networks(
            network=network,
            device_count=torch.cuda.device_count()
        )
            
        self.optimizers: typing.List[nn.Module] = get_gpu_optimizers(
            name=optimizer_name.lower(), 
            networks=self.networks,
            opt_params=optimizer_config, 
            devices=gpu_devices
        )
        
        self.loss_functions = get_gpu_losses(
            loss_function=loss_function,
            device_count=torch.cuda.device_count()
        )
    
    def train_batch(self, images, labels):
        """
        Trains network over a single batch of data
        """
        # splitting data into separate batches across different available devices
        
        # running predictions of seperate GPUs
        predictions = [
            self.networks[idx].forward(images[idx])
            for idx in range(len(self.gpu_devices))
        ]
        
        # calculating losses for each gpu device
        
        losses = [
            self.loss_functions[idx](predictions[idx], labels[idx])
            for idx in range(len(self.gpu_devices))
        ]
        
        # calculating and storing gradients on separate GPUs
        for loss in losses:
            loss.backward()
            
        # summing up gradients and given back to gpus
        # we do this before optimizer.step() as it does the actual
        # work of updating the parameters, while loss.backward() simply
        # computes the gradients.
        
        with torch.no_grad():
            param_idx = 0
            for param in self.networks[0].parameters():
                self.allreduce(
                    [
                        list(self.networks[network_idx].parameters())[param_idx].grad 
                        for network_idx in range(len(self.networks))
                    ]
                )
                param_idx += 1
                
        # updating parameters after computing gradients
        for device_idx in range(len(self.gpu_devices)):
            self.optimizers[device_idx].zero_grad()
            self.optimizers[device_idx].step()
            
        return torch.mean(sum(losses))
        
    def allreduce(self, gradients: typing.List[torch.Tensor]):
        """
        Aggregates computed gradients from different GPU's 
        into a single huge batch. Usually this method is called
        on the main GPU unit, which serves as a master.
    
        Args:
            - gradients - computed gradients from other GPUs
        """
        # copying data to the master gpu node

        for idx in range(1, len(gradients)):
            gradients[0][:] += gradients[idx][:].to(self.gpu_devices[0])

        # giving away data back to other nodes
        for idx in range(1, len(gradients)):
            gradients[idx][:] = gradients[0][:].to(self.gpu_devices[idx])
    
    def split_input_into_batches(self, imgs, labels):
        assert len(imgs) == len(labels)
        batches = []
        batch_size = len(labels) // len(self.gpu_devices)
        for idx in range(len(self.gpu_devices)):
            start = idx*batch_size
            end = start + batch_size 
            batches.append((imgs[start:end], labels[start:end]))
        return batches 

    def train(self, 
        input_images: typing.List[torch.Tensor], 
        input_labels: typing.List[torch.Tensor]
    ):
        if len(input_images.shape) != 4:
            raise ValueError("""invalid number of 
            dimensions: need 4, but have %s""" % input_images.shape[0])
            
        train_loss = []
        batches = self.split_input_into_batches(input_images, input_labels)

        for _ in range(self.max_epochs):
            
            for images, labels in batches:
                
                split_images = parallel.scatter(images, self.gpu_devices)
                split_labels = parallel.scatter(labels, self.gpu_devices)
                
                loss = self.train_batch(
                    images=split_images, 
                    labels=split_labels
                )
                torch.cuda.synchronize()
                train_loss.append(loss.item())
        return train_loss