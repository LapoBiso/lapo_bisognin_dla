import torch
from sklearn.metrics import classification_report
import numpy as np

import wandb
from omegaconf import DictConfig, OmegaConf
import argparse

import utils, data_setup
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments

def main(cfg: DictConfig):
    device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu')
    ds_dict = data_setup.load_data(cfg.dataset.name)
    ds_train = ds_dict['train']
    ds_val = ds_dict['validation']
    ds_test = ds_dict['test']
    model = AutoModelForSequenceClassification.from_pretrained(cfg.model.checkpoint, num_labels=2).to(device)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.checkpoint)
    tokenized_train = data_setup.set_tokenized_data(ds_train, tokenizer)
    tokenized_val = data_setup.set_tokenized_data(ds_val, tokenizer)
    tokenized_test = data_setup.set_tokenized_data(ds_test, tokenizer)
    data_collator = DataCollatorWithPadding(tokenizer = tokenizer)
    
    with wandb.init(project=cfg.project_name, name=cfg.run_name, config=OmegaConf.to_container(cfg,resolve=True)):
        
        training_args = TrainingArguments(
            output_dir = 'DLA-Lab2/checkpoints',
            learning_rate = cfg.opt.lr,
            per_device_train_batch_size = cfg.training.batch_size,
            per_device_eval_batch_size = cfg.training.batch_size,
            num_train_epochs = cfg.training.epochs,
            use_cpu = False,
            save_strategy = 'best',
            report_to = 'wandb',
            logging_strategy = 'steps',
            logging_steps = 1,
            do_eval = True,
            eval_strategy = 'epoch',
            load_best_model_at_end=True,
            metric_for_best_model='eval_loss',
            greater_is_better = False
            
        )

        trainer = Trainer(
            model = model,
            args = training_args,
            train_dataset = tokenized_train,
            data_collator = data_collator,
            eval_dataset = tokenized_val,
            compute_metrics = utils.compute_metrics,
            
        )

        trainer.train()
        test_metrics = utils.evaluate_cls_model(trainer, tokenized_test)
        print(test_metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     type=str,   default="DLA-Lab2/configs/config.yaml")
    parser.add_argument("--lr",         type=float, default=None)
    parser.add_argument("--batch_size", type=int,   default=None)
    parser.add_argument("--epochs",     type=int,   default=None)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    if args.lr         is not None: cfg.opt.lr         = args.lr
    if args.batch_size is not None: cfg.training.batch_size = args.batch_size
    if args.epochs     is not None: cfg.training.epochs     = args.epochs

    cfg.run_name = (
        f"lr={cfg.opt.lr}_"
        f"bs={cfg.training.batch_size}_"
        f"ep={cfg.training.epochs}"
    )

    main(cfg)