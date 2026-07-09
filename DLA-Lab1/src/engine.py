from sklearn.metrics import accuracy_score, classification_report
import torch.nn.functional as F
import torch.optim as opt
from data_setup import *
import torch.nn as nn
import copy
import wandb
import numpy as np
import hydra
from tqdm import tqdm

def train_step(cfg, model, dl, opt, loss, epoch, device = 'cpu'):
    model.train()
    losses = []
    for (x, y) in tqdm(dl, desc = f'train epoch {epoch}', leave = True):
        x = x.to(device)
        y = y.to(device)
        opt.zero_grad()
        logits = model(x)
        step_loss = loss(logits, y)
        step_loss.backward()
        opt.step()
        losses.append(step_loss.item())
    return np.mean(losses)


def validation_step(model, dl, loss, device = 'cpu'):
    
    model.eval()
    predictions = []
    gts = []
    losses = []
    with torch.no_grad():
        for (x, y) in tqdm(dl, desc = 'evaluating', leave = False):
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            pred = torch.argmax(logits, dim=1)
            losses.append(loss(logits, y).item())
            gts.append(y.cpu().numpy())
            predictions.append(pred.cpu().numpy())
    acc = accuracy_score(np.hstack(gts), np.hstack(predictions))
    cls_report = classification_report(np.hstack(gts), np.hstack(predictions), zero_division=0, digits=3)
    return acc, np.mean(losses), cls_report


def train(cfg, model, dl_train, dl_val, dl_test, device='cpu'):
    """
    train fine-tuned model
    """
    epochs = cfg.train.get('epochs')
    loss = hydra.utils.instantiate(cfg.loss)
    wandb.watch(model, log="all", log_freq=100)
    losses = []
    global_step = 1
    best_acc = 0
    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = hydra.utils.instantiate(cfg.opt, trainable)
    
    for epoch in tqdm(range(epochs)):
        loss_epoch = train_step(cfg, model, dl_train, optim, loss, epoch, device)
        val_acc, val_loss, cls_report = validation_step(model, dl_val, loss, device)
        losses.append(loss_epoch)
        wandb.log({"train/loss_epoch": loss_epoch,
                   "val/loss_epoch": val_loss, "val/acc": val_acc, "epoch": epoch+1,}, step=global_step)
        global_step += 1

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    test_acc, test_loss, cls_report = validation_step(model, dl_test, loss, device)
    wandb.log({"test/loss": test_loss, "test/accuracy": test_acc})
    return losses, cls_report

def extract_fts(model, dl, device):
    """
    use model to extract features from input data
    """
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for (x, y) in tqdm(dl):
            x = x.to(device)
            feat = model(x)
            features.append(feat.cpu())
            labels.append(y)
    features = torch.cat(features, axis=0)
    labels = torch.cat(labels, axis=0)
    return features, labels