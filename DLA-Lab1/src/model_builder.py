from torchvision.models import list_models, get_model
import torchvision.ops as ops
from data_setup import *
import torch.nn as nn

def build_baseline(cfg, device):
    """
    build baseline model based on configuration details
    """
    name = cfg.model.get('name')
    model = get_model(name, weights='DEFAULT')
    
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        fc_in = model.fc.in_features
        model.fc = nn.Identity()
        head_attr = 'fc'
    elif name == 'vgg19':
        fc_in = model.classifier[6].in_features
        model.classifier[6] = nn.Identity()
        head_attr = ('classifier', 6)
    elif name == 'mobilenet_v2':
        fc_in = model.classifier[1].in_features
        model.classifier[1] = nn.Identity()
        head_attr = ('classifier', 1)
    elif name == 'mobilenet_v3_small':
        fc_in = model.classifier[3].in_features
        model.classifier[3] = nn.Identity()
        head_attr = ('classifier', 3)
    elif name in ('convnext_base', 'convnext_tiny'):
        fc_in = model.classifier[2].in_features
        model.classifier[2] = nn.Identity()
        head_attr = ('classifier', 2)
    else:
        raise ValueError(f"Unrecognized head for {name}")
    
    model.to(device)
    return model, fc_in, head_attr


def build_head(cfg, model, fc_in):
    """
    add new model head based on config option
    """
    head = cfg.model.head.get("type")
    if head == "linear":
        model.fc = nn.Linear(in_features=fc_in, out_features=cfg.model.get('num_cls'))
    elif head == "MLP":
        hidden = list(cfg.model.head.get("hidden"))
        dims = hidden + [cfg.model.get('num_cls')]
        model.fc = ops.MLP(fc_in,dims)
    return

def freeze_layers(cfg, model):
    """
    freeze model layers training
    """
    layers = ["layer1"]
    freeze = cfg.model.get("freeze_lrs")
    if freeze == 0:
        return
    else:
        for i in layers:
            for p in getattr(model, i).parameters():
                p.requires_grad = False
        return


def build_ft_model(cfg, device):
    """
    build model for fine-tuning
    """
    model,fc_in, head_attr = build_baseline(cfg, device)
    build_head(cfg, model, fc_in)
    freeze_layers(cfg, model)
    model.to(device)
    return model
