from transformers import AutoModel,AutoModelForSequenceClassification
fintuned_model_name = f'{base_model_name}-finetuned-panx-{lang}/checkpoint-2502'
model = AutoModelForSequenceClassification.from_pretrained(fintuned_model_name).to(device)