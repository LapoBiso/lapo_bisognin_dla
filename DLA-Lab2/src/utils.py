import collections
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
import wandb

def count_labels(ds_dict):
    """
    shows labels distribution in dataset splits
    """
    for splits, ds in ds_dict.items():
        counts = collections.Counter(ds['label'])
        total = len(ds)
        for label, count in counts.items():
            name = 'pos' if label == 1 else 'neg'
            print(f'{splits} {name} ({label}): {count}')

def text_length(ds_dict):
    """
    shows text length distribution
    """
    ds_train = ds_dict['train']
    texts = ds_train['text']
    labels = ds_train['label']

    lengths = np.array([len(t.split()) for t in texts])
    max_l = lengths.max()
    min_l = lengths.min()
    mean_l = lengths.mean()
    print(f'max text length: {max_l}, min text length {min_l}, mean text length {mean_l:.2f}')

def freq_words_label(texts, labels, target_label, stop_words):
    words = []
    for t, l in zip(texts, labels):
        if l == target_label:
            words.extend(w for w in t.lower().split() if w not in stop_words)
    return collections.Counter(words).most_common(20)

def freq_words(ds_dict):
    """
    shows freq words corresponding to positive and negatives texts
    """
    texts = ds_dict['train'] ['text']
    labels = ds_dict['train']['label']
    sw = []
    for t in texts:
        sw.extend(t.lower().split())
    stop_words_count = collections.Counter(sw).most_common(50)
    stop_words = [word for word, count in stop_words_count]
    frequent_neg_words = freq_words_label(texts, labels, 0, stop_words)
    frequent_pos_words = freq_words_label(texts, labels, 1, stop_words)
    print(f'Frequent words form negative labels: {frequent_neg_words}\n Frequent words form positive labels: {frequent_pos_words} ')

def compute_metrics(eval_pred):
    """
    trainer metrics definition for evaluation
    """
    logits, labels = eval_pred

    preds = np.argmax(logits, axis = -1)

    return {
        'accuracy': accuracy_score(labels, preds),
        'f1_score': f1_score(labels, preds),
        'precision': precision_score(labels, preds),
        'recall': recall_score(labels, preds)
    }

def evaluate_cls_model(trainer, ds_tokenized):
    """
    evaluates trained model on test data
    """
    output = trainer.predict(ds_tokenized)
    preds = np.argmax(output.predictions, axis=-1)
    test_labels = output.label_ids
    print("TEST:")
    wandb.log({"test/accuracy": (preds == test_labels).mean()})
    return classification_report(test_labels, preds)
