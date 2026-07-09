import os
import torch
from torchvision.datasets import GTSRB
import data_setup, engine, model_builder, utils
import torchvision.transforms.v2 as T 
import pandas as pd
import matplotlib.pyplot as plt

import hydra
from omegaconf import DictConfig


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model_builder.build_ft_model(cfg, device)
    ds_path = os.path.join(cfg.dataset.get('ds_path'), cfg.dataset.get('name'))
    download = not data_setup.is_dataset_present(ds_path)
    transform = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])
    ds_train = GTSRB(cfg.dataset.get('ds_path'), split="train", download=True, transform=transform)

    if cfg.experiments.get('plot_cls'):
        classes = utils.sample_classes(ds_train)
        utils.plot_dataset(classes, cfg.experiments.get('output_path'))
        
    if cfg.experiments.get('plot_distribution'):
        labels = torch.tensor([label for _,label in ds_train])
        utils.data_distribution(labels)
        table = pd.DataFrame([(cls, im.shape[1], im.shape[2]) for (im, cls) in ds_train], 
                             columns=['CLS', 'HEIGHT', 'WIDTH'])
        axes = table.hist(figsize=(12, 8))
        fig = axes.flatten()[0].get_figure()
        fig.savefig(os.path.join(cfg.experiments.get('output_path'),'cls_distribution.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(table.describe())



if __name__ == '__main__':
    main()