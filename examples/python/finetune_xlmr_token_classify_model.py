import torch.nn as nn
import torch
import pandas as pd
import numpy as np
from transformers import XLMRobertaConfig, AutoConfig, AutoTokenizer, TrainingArguments, Trainer, \
    DataCollatorForTokenClassification
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaPreTrainedModel
from datasets import load_dataset
from seqeval.metrics import f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class XLMRobertaForTokenClassification(RobertaPreTrainedModel):
    config_class = XLMRobertaConfig

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        # load base model body
        self.roberta_model_body = RobertaModel(config)

        # add classification head
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier_head = nn.Linear(config.hidden_size, config.num_labels)

        # load init base model weights as initial weights of new model
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
        # use base model outputs as inputs of newly classifier head
        base_model_outputs = self.roberta_model_body(input_ids=input_ids, attention_mask=attention_mask,
                                                     token_type_ids=token_type_ids, **kwargs)
        # Feed roberta model body's outputs to classifier
        dropout_output = self.dropout(base_model_outputs[0])
        logits = self.classifier_head(dropout_output)

        # calculate loss
        loss = None
        if labels is not None:
            loss_fac = nn.CrossEntropyLoss()
            loss = loss_fac(logits.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=base_model_outputs.hidden_states,
                                     attentions=base_model_outputs.attentions)


def token_and_align_labels(batch, xlmr_tokenizer):
    tokenized_inputs = xlmr_tokenizer(batch['tokens'], truncation=True, is_split_into_words=True)

    labels = []
    for idx, label in enumerate(batch['ner_tags']):
        word_ids = tokenized_inputs.word_ids(batch_index=idx)
        previous_word_idx = None
        label_ids = []
        for word_id_idx in word_ids:
            if word_id_idx is None or word_id_idx == previous_word_idx:
                label_ids.append(-100)
            elif word_id_idx != previous_word_idx:
                label_ids.append(label[word_id_idx])

            previous_word_idx = word_id_idx
        labels.append(label_ids)
    tokenized_inputs['labels'] = labels
    return tokenized_inputs


def encode_panx_dataset(corpus):
    return corpus.map(token_and_align_labels, batched=True, remove_columns=['langs', 'tokens', 'ner_tags'])


class Utils(object):

    def __init__(self, tags):
        self.tags = tags

    def idx2tag(self):
        return {idx: tag for idx, tag in enumerate(self.tags)}

    def tag2idx(self):
        return {tag: idx for idx, tag in enumerate(self.tags)}

    def align_preds(self, predictions, label_ids):
        preds = np.argmax(predictions, axis=2)
        batch_size, seq_len = preds.shape
        label_list, pred_list = [], []

        for batch_idx in range(batch_size):
            ex_labels, ex_preds = [], []
            for seq_idx in range(seq_len):
                if label_ids[batch_idx][seq_idx] != -100:
                    ex_labels.append(self.idx2tag()[label_ids[batch_idx][seq_idx]])
                    ex_preds.append(preds[batch_idx][seq_idx])
            label_list.append(ex_labels)
            pred_list.append(ex_preds)

        return label_list, pred_list

    def compute_metrics(self, eval_pred):
        y_pred, y_true = self.align_preds(eval_pred.predictions, eval_pred.label_ids)
        return {"f1": f1_score(y_true, y_pred)}


def train(num_epochs=3, batch_size=24):
    base_model_name = 'xlm_roberta_base'
    lang = 'zh'
    xlmr_config = AutoConfig.from_pretrained(base_model_name)
    xlmr_tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    dataset = load_dataset('xtreme', name=f'PAN-X.{lang}')
    encoded_dataset = encode_panx_dataset(dataset)
    print('encoded_dataset', encoded_dataset)
    print('encoded_dataset features', encoded_dataset.features)

    logging_steps = len(encoded_dataset) // batch_size
    model_name = f'{base_model_name}-finetuned-panx-{lang}'
    training_args = TrainingArguments(
        output_dir=model_name, log_level="error", num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy='epoch',
        save_steps=1e6, weight_decay=0.01, disable_tqdm=False,
        logging_steps=logging_steps, push_to_hub=False
    )

    # padding to max sequence len in batch
    data_collator = DataCollatorForTokenClassification(xlmr_tokenizer)

    def model_init():
        return XLMRobertaForTokenClassification.from_pretrained(base_model_name, config=xlmr_config).to(device)

    tags = dataset['train'].features['ner_tags'].feature
    utils = Utils(tags)

    trainer = Trainer(model_init=model_init, args=training_args, data_collator=data_collator,
                      compute_metrics=utils.compute_metrics, train_dataset=encoded_dataset['train'],
                      eval_dataset=encoded_dataset['validation'])
    print(f'Begining to train: {base_model_name}')
    trainer.train()
    print(f'End to train: {base_model_name}')


if __name__ == "__main__":
    train()
