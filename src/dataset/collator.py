import torch
from transformers import DataCollatorForSeq2Seq

class TSACollator():
    def __init__(self, tokenizer=None, padding=True, **kwargs):
        self.tokenizer = tokenizer
        self.label_pad_token_id = kwargs['label_pad_token_id']
        self.seq2seq_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer, 
            padding=padding, 
            **kwargs
        )

    def __call__(self, features, return_tensors=None):
        bs = len(features)
        batch_doc_ids = []
        batch_position_ids = []
        batch_aux_labels = []
        for feature in features:
            doc_ids = feature.pop('doc_ids')
            position_ids = feature.pop('position_ids')
            batch_aux_labels.append(feature.pop('aux_labels'))
            batch_doc_ids.append(doc_ids)
            batch_position_ids.append(position_ids)

        data = self.seq2seq_collator(features, return_tensors)
        max_doc_len = data['input_ids'].shape[1]
        padding_batch_doc_ids = torch.zeros(bs, max_doc_len, dtype=torch.long)
        for i in range(bs):
            padding_batch_doc_ids[i, :len(batch_doc_ids[i])] = torch.LongTensor(batch_doc_ids[i])

        padding_batch_position_ids = []
        for i in range(bs):
            last_position_id = batch_position_ids[i][-1]
            post_len = max_doc_len - len(batch_position_ids[i])
            temp_position_ids = batch_position_ids[i] + [i for i in range(last_position_id+1, last_position_id + 1 + post_len)]
            padding_batch_position_ids.append(temp_position_ids)
        
        padding_batch_position_ids = torch.LongTensor(padding_batch_position_ids)
        data['doc_ids'] = padding_batch_doc_ids
        data['position_ids'] = padding_batch_position_ids
        data['batch_aux_labels'] = batch_aux_labels
        data.pop("lengths")
        return data
