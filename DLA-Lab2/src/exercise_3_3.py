import numpy as np
import torch


import os
from omegaconf import DictConfig, OmegaConf
import argparse

import data_setup, engine, utils
from transformers import CLIPModel, CLIPProcessor

def main(cfg: DictConfig):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ft_path = cfg.dataset.ft_path
    model     = CLIPModel.from_pretrained(cfg.model.checkpoint).to(device)
    processor = CLIPProcessor.from_pretrained(cfg.model.checkpoint)
    ds_dict  = data_setup.load_data(cfg.dataset.name)
    ds_train = ds_dict['train']
    if os.path.exists(ft_path):
        images_ft = torch.load(ft_path, map_location=device)
    else:
        images_ft = engine.extract_clip_ft(cfg, ds_train, processor, model, device = device)
        torch.save(images_ft, ft_path)
    text = "Black dog"
    prompt_ft = engine.get_text_features(text, processor, model, device = device)
    topk = engine.retrieve_images(cfg, prompt_ft, images_ft)
    images = []
    for rank, idx in enumerate(topk):
        image = ds_train[idx.item()]['image']
        image.save(f'DLA-Lab2/retrieval_results/rank_{rank}.jpg')
        images.append(image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     type=str,   default="DLA-Lab2/configs/config.yaml")
    parser.add_argument("--checkpoint",  type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--dataset", type=str,   default="jxie/flickr8k")

    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    if args.checkpoint is not None: cfg.model.checkpoint = args.checkpoint
    if args.dataset     is not None: cfg.dataset.name    = args.dataset



    main(cfg)