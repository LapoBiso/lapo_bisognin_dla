import torch
import data_setup

CAPTION_COLS = ['caption_0', 'caption_1', 'caption_2', 'caption_3', 'caption_4']

def extract_ft(ds_dict, extractor):
    """
    execute feature extraction for classification task
    """
    ds_train = ds_dict['train']
    ds_test = ds_dict['test']
    ds_val = ds_dict['validation']
    feats = extractor(list(ds_train['text']), return_tensors = 'pt')
    test_feats = extractor(list(ds_test['text']), return_tensors = 'pt')
    val_feats = extractor(list(ds_val['text']), return_tensors = 'pt')
    feats, tr_labels = data_setup.set_classifier_data(ds_train, feats)
    test_feats, test_labels = data_setup.set_classifier_data(ds_test, test_feats)
    val_feats, val_labels = data_setup.set_classifier_data(ds_val, val_feats)
    return feats, tr_labels, val_feats, val_labels, test_feats, test_labels

def extract_clip_ft(cfg, ds_dict, processor, model, device):
    images_ft = []
    captions_ft = []
    batch_size = cfg.training.batch_size
    for i in range(0, len(ds_dict), batch_size):
        batch = ds_dict.select(range(i, min(i + batch_size, len(ds_dict))))

        input = processor(
            images = list(batch['image']),
            return_tensors = 'pt',
            padding = True,
            truncation = True
        ).to(device)

        with torch.no_grad():
            im_ft = model.get_image_features(pixel_values = input['pixel_values']).pooler_output

        images_ft.append(im_ft)
    return torch.vstack(images_ft)
        

def get_text_features(prompt, processor, model, device):
    inputs = processor(
        text=prompt,
        return_tensors='pt',
        padding=True,
        truncation=True
    ).to(device)
    
    with torch.no_grad():
        text_ft = model.get_text_features(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        ).pooler_output

    return text_ft

def retrieve_images(cfg, query, images):
    query_norm = query / query.norm(dim = -1, keepdim = True)
    images_norm = images / images.norm(dim = -1, keepdim = True)
    similarity = query_norm @ images_norm.T
    return similarity.topk(k = cfg.exercise_3.k, dim = -1).indices[0]
