from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from matplotlib import pyplot as plt
import torch

local_model_ckpt = "/root/hugging_model_hub/distilbert-base-uncased"
local_dataset = "/root/hugging_model_hub/datasets/emotion"
tokenizer = AutoTokenizer.from_pretrained(local_model_ckpt)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained(local_model_ckpt).to(device)

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

def extract_hidden_state(batch):
    inputs = {k: v.to(device) for k, v in batch.items() if k in tokenizer.model_input_names}

    with torch.no_grad():
        last_hidden_state = model(**inputs).last_hidden_state

    return {"hidden_state": last_hidden_state[:,0].cpu().numpy()}

emotion_datasets = load_dataset(local_dataset)
emotions_encoded = emotion_datasets.map(tokenize, batched=True, batch_size=None)
emotions_encoded.set_format("torch", columns=["input_ids", "attention_mask", "label"])

emotions_hidden = emotions_encoded.map(extract_hidden_state, batched=True)
print(emotions_hidden['train'].column_names, emotions_hidden['train'].shape)

import numpy as np
import pandas as pd
X_train = np.array(emotions_hidden["train"]["hidden_state"])
X_valid = np.array(emotions_hidden["validation"]["hidden_state"])
y_train = np.array(emotions_hidden["train"]["label"])
y_valid = np.array(emotions_hidden["validation"]["label"])

# from umap import UMAP
# from sklearn.preprocessing import MinMaxScaler
# # Scale features to [0,1] range
# X_scaled = MinMaxScaler().fit_transform(X_train)
# # Initialize and fit UMAP
# mapper = UMAP(n_components=2, metric="cosine").fit(X_scaled)
# # Create a DataFrame of 2D embeddings
# df_emb = pd.DataFrame(mapper.embedding_, columns=["X", "Y"])
# df_emb["label"] = y_train
# print(df_emb.head())

# fig, axes = plt.subplots(2, 3, figsize=(7, 5))
# axes = axes.flatten()
# cmaps = ["Greys", "Blues", "Oranges", "Reds", "Purples", "Greens"]
labels = emotion_datasets["train"].features["label"].names
# for i, (label, cmap) in enumerate(zip(labels, cmaps)):
#     df_emb_sub = df_emb.query(f"label == {i}")
#     axes[i].hexbin(df_emb_sub["X"], df_emb_sub["Y"], cmap=cmap,
#                    gridsize=20, linewidths=(0,))
#     axes[i].set_title(label)
#     axes[i].set_xticks([]), axes[i].set_yticks([])
# plt.tight_layout()
# plt.show()


#simple classifer
# from sklearn.linear_model import LogisticRegression
# lr_clf = LogisticRegression(max_iter=3000)
# lr_clf.fit(X_train, y_train)
# lr_score=lr_clf.score(X_valid, y_valid)
# print('lr_score:', lr_score)

# #dummy classifer
# from sklearn.dummy import DummyClassifier
# dummy_clf = DummyClassifier(strategy="most_frequent")
# dummy_clf.fit(X_train, y_train)
# dummy_score=dummy_clf.score(X_valid, y_valid)
# print('dummy_score:', dummy_score)

#investigate performance of model by confusion matrix
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
def plot_confusion_matrix(y_preds, y_true, labels):
    matrix=confusion_matrix(y_true, y_preds, normalize="true")
    figure, axies = plt.subplots(figsize=(6,6))
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", ax=axies, colorbar=False)
    plt.title("Normalize confusion matrix")
    plt.show()

# y_preds=lr_clf.predict(X_valid)
# plot_confusion_matrix(y_preds, y_valid, labels)

# fine-tune with transformers
from transformers import AutoModelForSequenceClassification


num_labels=6
model=AutoModelForSequenceClassification.from_pretrained(local_model_ckpt, num_labels=num_labels).to(device)

#define model performance metrics using in trainning
from sklearn.metrics import accuracy_score, f1_score
from transformers import Trainer, TrainingArguments, Trainer

def compute_metrics(pred):
     labels = pred.label_ids
     preds = pred.predictions.argmax(-1)
     f1 = f1_score(labels, preds, average="weighted")
     acc = accuracy_score(labels, preds)
     return {"accuracy": acc, "f1": f1}

#define hyper parameters
batch_size=64
logging_steps=len(emotions_encoded["train"]) // batch_size
model_name=f"{local_model_ckpt}-finetune-emotion"
train_args=TrainingArguments(output_dir=model_name,
                              num_train_epochs=2,
                              learning_rate=2e-5,
                              per_device_train_batch_size=batch_size,
                              per_device_eval_batch_size=batch_size,
                              weight_decay=0.01,
                              eval_strategy="epoch",
                              disable_tqdm=False,
                              logging_steps=logging_steps,
                              push_to_hub=False,
                              log_level="error"
)
trainer = Trainer(model=model, args=train_args, compute_metrics=compute_metrics,
                  train_dataset=emotions_encoded["train"],
                  eval_dataset=emotions_encoded["validation"],
                  tokenizer=tokenizer)
trainer.train()

# validate performance
preds_output = trainer.predict(emotions_encoded["validation"])
preds_output.metrics
y_preds = np.argmax(preds_output.predictions, axis=1)
plot_confusion_matrix(y_preds, y_valid, labels)