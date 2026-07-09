from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

from omegaconf import DictConfig, OmegaConf
import argparse

import utils, data_setup, model_builder, engine

def main(cfg: DictConfig):
    ds_dict = data_setup.load_data(cfg.dataset.name)

    if cfg.exercise_1.read_data:
        utils.count_labels(ds_dict)
        print('\n')
        utils.freq_words(ds_dict)
        print('\n')
        utils.text_length(ds_dict)

    if cfg.exercise_1.build_classifier:
        model, tokenizer, extractor = model_builder.build_baseline(cfg)
        feats, tr_labels, val_feats, val_labels, test_feats, test_labels = engine.extract_ft(ds_dict, extractor)
        param_grid = {
            'C':            [0.01, 0.1, 1, 10],
            'kernel':       ['linear', 'rbf'],
        }
        print("start grid")
        grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='f1_macro')
        grid.fit(feats, tr_labels)
        print(f"Best model parameters:{grid.best_params_}")

        val_preds = grid.best_estimator_.predict(val_feats)
        print("VALIDATION: ")
        print(classification_report(val_labels, val_preds))

        test_preds = grid.best_estimator_.predict(test_feats)
        print("TEST: ")
        print(classification_report(test_labels, test_preds))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="DLA-Lab2/configs/config.yaml")
    parser.add_argument("--read_data", type  = str, default = False)
    parser.add_argument("--build_classifier", type  = str, default = True)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)

    if args.read_data is not None: cfg.exercise_1.read_data = args.read_data
    if args.read_data is not None: cfg.exercise_1.build_classifier = args.build_classifier

    main(cfg)