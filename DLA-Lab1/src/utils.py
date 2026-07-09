import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve
import numpy as np
import torch
import os
from torch.nn.functional import normalize
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import seaborn as sns

def data_distribution(ds_train):
    """
    counts dataset labels distribution
    """
    counts = torch.bincount(ds_train)
    print(f'Min counts: {counts.min().item()} (Class: {counts.argmin().item()})')
    print(f'Max counts: {counts.max().item()} (Class: {counts.argmax().item()})')
    print(f'Ratio:{counts.max().item()/counts.min().item()}')

def sample_classes(ds_train):
    """
    sample one element for each class in the dataset
    """
    unique_samples = {}
    for item in ds_train:
        image, label = item[0], item[1] 
        if label not in unique_samples:
            unique_samples[label] = (item)
        if len(unique_samples) == 43:
            break
    return unique_samples

def plot_dataset(image_samples, path):
    """
    plot grid of dataset images
    """
    output_dir = path
    output_path = os.path.join(output_dir, "dataset_classes_grid.png")
    fig, axes = plt.subplots(5, 9, figsize=(20, 12))
    axes = axes.flatten()
    for i in range(45):
        if i < len(image_samples):
            img, label = image_samples[i][0], image_samples[i][1]
            if hasattr(img, 'numpy'):
                img = img.numpy().transpose(1, 2, 0)
                img = np.clip(img, 0, 1)
            if len(img.shape) == 2 or img.shape[2] == 1:
                axes[i].imshow(img, cmap='gray')
            else:
                axes[i].imshow(img)
            axes[i].set_title(f"Class: {label}", fontsize=10)
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close(fig)


def plot_loss(loss):
    plt.figure(figsize=(8,5))
    plt.plot(loss)
    plt.savefig('DLA-Lab1/Lab1_outputs/figures/')

def retrieval_fts(d, q, device):
    """
    calculate cosine similarities between query features and dataset features
    """
    d = d.to(device)
    q = q.to(device)
    d_norm = normalize(d)
    q_norm = normalize(q)
    scores = (q_norm @ d_norm.T).cpu()
    return scores

def print_similarities(cfg, labels_query, sorted_idx, sorted_labels, sorted_scores, cls, K):
    """
    given class cls print top similarities along dataset elements
    """
    print(f"=== Test {cls} — true label: {labels_query[cls].item()} - backbone: {cfg.model.get('name')}===")
    print(f"{'rank':>4} {'train_idx':>10} {'label':>6} {'score':>8}")
    for rank in range(K):
        idx = sorted_idx[cls, rank].item()
        score = sorted_scores[cls, rank].item()
        label = sorted_labels[cls, rank].item()
        print(f"{rank:>4} {idx:>10} {label:>6} {score:>8.4f}")


def ap_per_query(scores, labels_train, cls, return_curve=False):
    """
    compute average precision for a single cls element
    """
    y_true = (labels_train == cls).cpu().numpy().astype(int)
    y_score = scores.cpu().numpy()
    ap = average_precision_score(y_true, y_score)
    if return_curve:
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        return ap, precision, recall
    return ap

def interp_precision_at_recall(precision, recall, recall_grid):
    """
    interpolate a precision-recall curve onto a common recall grid using the
    standard "precision envelope" (max precision achievable at recall >= r)
    """
    order = np.argsort(recall)
    r = recall[order]
    p = precision[order]
    envelope = np.maximum.accumulate(p[::-1])[::-1]
    idx = np.searchsorted(r, recall_grid, side='left')
    idx = np.clip(idx, 0, len(envelope) - 1)
    return envelope[idx]

def class_ap(scores, labels_query, labels_train, cls, return_curve=False, recall_grid=None):
    """
    compute elements of class cls average precision
    """
    mask = labels_query == cls
    idxs = torch.where(mask)[0]
    if return_curve:
        if recall_grid is None:
            recall_grid = np.linspace(0, 1, 101)
        aps = []
        interp_precisions = []
        for i in idxs:
            ap, precision, recall = ap_per_query(scores[i], labels_train, cls, return_curve=True)
            aps.append(ap)
            interp_precisions.append(interp_precision_at_recall(precision, recall, recall_grid))
        mean_precision = np.mean(interp_precisions, axis=0)
        return sum(aps) / len(aps), recall_grid, mean_precision
    aps = [ap_per_query(scores[i], labels_train, cls)
           for i in idxs]
    return sum(aps) / len(aps)


def plot_pr_curves(scores, labels_query, labels_train, save_path):
    """
    plot top 5 and worst 5 class precision recall curve, averaged per-query
    over a common recall grid (macro-average, consistent with the AP metric)
    """
    num_classes = int(labels_query.max().item()) + 1
    recall_grid = np.linspace(0, 1, 101)

    aps_per_class = []
    for c in range(num_classes):
        ap, recall, precision = class_ap(scores, labels_query, labels_train, c,
                                          return_curve=True, recall_grid=recall_grid)
        aps_per_class.append((c, ap, recall, precision))

    aps_per_class.sort(key=lambda x: x[1])
    worst = aps_per_class[:5]
    best = aps_per_class[-5:]

    fig, (ax_best, ax_worst) = plt.subplots(1, 2, figsize=(14, 6))
    cmap_best = plt.get_cmap('Greens')
    cmap_worst = plt.get_cmap('Reds')

    for idx, (cls, ap, recall, precision) in enumerate(best):
        ax_best.plot(recall, precision, linewidth=2,
                     color=cmap_best(0.4 + 0.12 * idx),
                     label=f'cls {cls} (AP={ap:.3f})')

    ax_best.set_xlabel('Recall')
    ax_best.set_ylabel('Precision')
    ax_best.set_title('Top 5 classes by AP')
    ax_best.grid(alpha=0.3)
    ax_best.set_xlim([0, 1])
    ax_best.set_ylim([0, 1.05])
    ax_best.legend(loc='lower left', fontsize=9)

    for idx, (cls, ap, recall, precision) in enumerate(worst):
        ax_worst.plot(recall, precision, linewidth=2,
                      color=cmap_worst(0.4 + 0.12 * idx),
                      label=f'cls {cls} (AP={ap:.3f})')
    
    ax_worst.set_xlabel('Recall')
    ax_worst.set_ylabel('Precision')
    ax_worst.set_title('Bottom 5 classes by AP')
    ax_worst.grid(alpha=0.3)
    ax_worst.set_xlim([0, 1])
    ax_worst.set_ylim([0, 1.05])
    ax_worst.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def mean_ft(features, labels):
    """
    compute mean feature map for each class
    """
    num_cls = labels.max().item() + 1
    D = features.shape [1]
    sums = torch.zeros(num_cls, D, device=features.device)
    counts = torch.zeros(num_cls, device=features.device)
    sums.index_add_(0, labels, features)
    counts.index_add_(0, labels, torch.ones_like(labels, dtype=torch.float))
    return sums/counts.unsqueeze(1)

def nearest_mean_classifier(query_features, class_means):
    """
    find nearest feature map to query features using cosine similarity
    """
    q_norm = F.normalize(query_features, dim=1)
    p_norm = F.normalize(class_means, dim=1)
    sims = q_norm @ p_norm.T
    return sims.argmax(dim=1) 

def plot_hm_conf_mat(labels_query, preds, path):
    """
    plot class prediction heatmap made by nmc
    """
    group_order = (
    list(range(0, 9))# speed limits
    + list(range(9, 18))# prohibitions
    + list(range(18, 32))# warnings
    + list(range(32, 41))# directions
    + list(range(41,43))# rule end
    + [12, 13, 14, 17]# unique
    )

    cm = confusion_matrix(labels_query, preds, labels=group_order)
    sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, sum, out = np.zeros_like(cm, dtype=float), where = sum != 0)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm_norm, ax=ax, cmap='Blues', square=True,
                xticklabels=group_order, yticklabels=group_order)
    group_boundaries = [9, 17, 31, 40, 42]
    for b in group_boundaries:
        ax.axhline(b, color='red', linewidth=0.5)
        ax.axvline(b, color='red', linewidth=0.5)
    ax.set_title('Confusion matrix (rows=true, cols=pred), grouped by semantic family')
    if path is not None:
        fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
