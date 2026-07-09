from transformers import AutoTokenizer, AutoModel, pipeline

def build_baseline(cfg):
    model = AutoModel.from_pretrained(cfg.model.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.checkpoint)
    extractor = pipeline(cfg.model.task, model = model, tokenizer = tokenizer)
    return model, tokenizer, extractor

