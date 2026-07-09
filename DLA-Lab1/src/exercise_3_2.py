import torch
import data_setup, engine, model_builder, utils
from torchvision import transforms

import os
import hydra
from omegaconf import DictConfig, OmegaConf

import matplotlib.pyplot as plt
import argparse

@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    K = cfg.experiments.get('K')
    cls = cfg.experiments.get('cls')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _, _ = model_builder.build_baseline(cfg, device)
    ds_path = os.path.join(cfg.dataset.get('ds_path'), cfg.dataset.get('name'))
    dl_train, _, dl_test = data_setup.build_std_dataloader(cfg, ds_path) 
    features_tr, labels_tr = engine.extract_fts(model, dl_train, device)
    features_query, labels_query = engine.extract_fts(model, dl_test, device)
    cs = utils.retrieval_fts(features_tr, features_query, device)
    sorted_scores, sorted_idx = cs.sort(axis=1, descending=True)
    sorted_labels = labels_tr[sorted_idx]
    
    if cfg.experiments.get('print_similarities'):
        utils.print_similarities(cfg, labels_query, sorted_idx, 
                                 sorted_labels,sorted_scores, cls, K)
    
    if cfg.experiments.get('mAP'):
        mAP = 0
        num_cls = cfg.model.get('num_cls')
        for i in range(num_cls):
            cls = i
            precision = utils.class_ap(cs, labels_query, labels_tr,  cls)
            mAP += precision
        mAP = mAP/num_cls
        print(f'\n\nmAP using {cfg.model.get('name')} as backbone is: {mAP}')
    
    if cfg.experiments.get('pr_curve'):
        out_pr_path = os.path.join(cfg.experiments.get('output_path'), "pr_curve")
        utils.plot_pr_curves(cs, labels_query, labels_tr, out_pr_path)
    
    mean_ft = utils.mean_ft(features_tr, labels_tr)
    preds = utils.nearest_mean_classifier(features_query, mean_ft)
    accuracy = (preds == labels_query).float().mean().item()
    print(f'nearest mean classifier accuracy using {cfg.model.get('name')} as backbone is: {accuracy}')

    if cfg.experiments.get('heatmap'):
        out_hm_path = os.path.join(cfg.experiments.get('output_path'), "heatmap")
        utils.plot_hm_conf_mat(labels_query, preds, out_hm_path)

if __name__== "__main__":
    main()
    