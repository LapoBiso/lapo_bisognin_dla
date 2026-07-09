import gradio as gr
import exercise_3_3 
import torch
from sklearn.metrics import classification_report

import os
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
import argparse

import utils, data_setup, model_builder, engine
from datasets import load_dataset, get_dataset_split_names
from transformers import CLIPModel, CLIPProcessor
_cfg = OmegaConf.load("DLA-Lab2/configs/config.yaml")

def image_retrieval(text, cfg = _cfg):
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
        
    prompt_ft = engine.get_text_features(text, processor, model, device = device)
    topk = engine.retrieve_images(cfg, prompt_ft, images_ft)
    images = []
    for rank, idx in enumerate(topk):
        image = ds_train[idx.item()]['image']
        image.save(os.path.join(cfg.exercise_3.ret_path, f"rank_{rank}.jpg"))
        images.append(image)
    return images

with gr.Blocks() as demo:
    gr.Markdown(
    """
    # Hello!
    ### Type below your prompt, click the button and then wait for retrieved images
    """)
    inp = gr.Textbox(placeholder="Black dog")
    out = gr.Gallery()
    search = gr.Button("Search")
    search.click(image_retrieval, inputs = inp, outputs = out)

demo.launch(share=True)
