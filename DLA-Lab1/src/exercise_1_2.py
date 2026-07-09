import torch
import data_setup, engine, model_builder, utils
from torchvision import transforms
from sklearn.svm import SVC
from sklearn.metrics import classification_report

import wandb
import os
import hydra
from omegaconf import DictConfig, OmegaConf

import matplotlib.pyplot as plt
import argparse

@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model,_,_ = model_builder.build_baseline(cfg, device)
    ds_path = os.path.join(cfg.dataset.get('ds_path'), cfg.dataset.get('name'))
    dl_train, dl_val, dl_test = data_setup.build_std_dataloader(cfg, ds_path)
    tr_ft, tr_cls = engine.extract_fts(model, dl_train, device)
    test_ft, test_cls = engine.extract_fts(model, dl_test, device)
    svc = SVC(kernel='linear')
    svc.fit(tr_ft, tr_cls)
    print(classification_report(svc.predict(test_ft), test_cls))

if __name__== "__main__":
    main()