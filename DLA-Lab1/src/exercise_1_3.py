import torch
import data_setup, engine, model_builder

import os
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with wandb.init(project=cfg.wandb.project, name=cfg.wandb.run_name, config=cfg_dict) as run:
        ds_path = os.path.join(cfg.dataset.get('ds_path'), cfg.dataset.get('name'))
        model = model_builder.build_ft_model(cfg, device)
        dl_train, dl_val, dl_test = data_setup.build_std_dataloader(cfg, ds_path)
        losses, cls_report = engine.train(cfg, model, dl_train, dl_val, dl_test, device)

if __name__== "__main__":
    main()