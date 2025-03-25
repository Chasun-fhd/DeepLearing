import pandas as pd
from datasets import load_dataset, tqdm
from nltk.tokenize import sent_tokenize
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import torch
import evaluate
from matplotlib import pyplot as plt


def lookup_dataset():
    dataset_samsum = load_dataset("samsum")
    split_lengths = [len(dataset_samsum[split]) for split in
                     dataset_samsum]
    print(f"Split lengths: {split_lengths}")
    print(f"Features: {dataset_samsum['train'].column_names}")
    print("\nDialogue:")
    print(dataset_samsum["test"][0]["dialogue"])
    print("\nSummary:")
    print(dataset_samsum["test"][0]["summary"])


# lookup_dataset()


device = "cuda" if torch.cuda.is_available() else "cpu"
model_ckpt = "google/pegasus-cnn_dailymail"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)


def chunks(list_of_elements, batch_size):
    """Yield successive batch-sized chunks from list_of_elements."""
    for i in range(0, len(list_of_elements), batch_size):
        yield list_of_elements[i: i + batch_size]


def evaluate_summaries_pegasus(dataset, metric, model, tokenizer,
                               batch_size=16, device=device,
                               column_text="article",
                               column_summary="highlights"):
    article_batches = list(chunks(dataset[column_text], batch_size))
    target_batches = list(chunks(dataset[column_summary],
                                 batch_size))
    for article_batch, target_batch in tqdm(
            zip(article_batches, target_batches),
            total=len(article_batches)):
        inputs = tokenizer(article_batch, max_length=1024,
                           truncation=True,
                           padding="max_length", return_tensors="pt")
        summaries = model.generate(input_ids=inputs["input_ids"].to(device),

                                   attention_mask=inputs["attention_mask"].to(device),
                                   length_penalty=0.8, num_beams=8,
                                   max_length=128)
    decoded_summaries = [tokenizer.decode(s,
                                          skip_special_tokens=True,
                                          clean_up_tokenization_spaces=True)
                         for s in summaries]
    decoded_summaries = [d.replace("<n>", " ") for d in
                         decoded_summaries]
    metric.add_batch(predictions=decoded_summaries,
                     references=target_batch)
    score = metric.compute()
    return score


def three_sentence_summary(text):
    return "\n".join(sent_tokenize(text)[:3])


def evaluate_summaries_baseline(dataset, metric, column_text="article", column_summary="highlights"):
    summaries = [three_sentence_summary(text) for text in dataset[column_text]]
    metric.add_batch(predictions=summaries, references=dataset[column_summary])
    score = metric.compute()
    return score


def data_distribution(dataset, tokenizer):
    d_len = [len(tokenizer.encode(s)) for s in dataset["train"]
    ["dialogue"]]
    s_len = [len(tokenizer.encode(s)) for s in dataset["train"]
    ["summary"]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)
    axes[0].hist(d_len, bins=20, color="C0", edgecolor="C0")
    axes[0].set_title("Dialogue Token Length")
    axes[0].set_xlabel("Length")
    axes[0].set_ylabel("Count")
    axes[1].hist(s_len, bins=20, color="C0", edgecolor="C0")
    axes[1].set_title("Summary Token Length")
    axes[1].set_xlabel("Length")
    plt.tight_layout()
    plt.show()


def convert_examples_to_features(examples_batch):
    input_encodings = tokenizer(examples_batch["dialogue"], max_length=1024, truncation=True)
    """
        一些模型在解码器输入中需要特殊的标记，因此区分编码器和解码器输入的分词非常重要。在with语句（称为上下文管理器）中，分词器知道它正在为解码器进行分词，并可以相应地处理序列。
    """
    with tokenizer.as_target_tokenizer():
        target_encodings = tokenizer(examples_batch["summary"], max_length=128, truncation=True)
        return {
            "input_ids": input_encodings["input_ids"],
            "attention_mask": input_encodings["attention_mask"],
            "labels": target_encodings["input_ids"],
        }


def train_entrance():
    """
    Finetune pegasus
    :return:
    """
    dataset_samsum = load_dataset("samsum")

    rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]

    data_distribution(dataset_samsum, tokenizer=tokenizer)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt).to(device)

    split_lengths = [len(dataset_samsum[split]) for split in dataset_samsum]
    print(f"Split lengths: {split_lengths}")
    print(f"Features: {dataset_samsum['train'].column_names}")
    print("\nDialogue:")
    print(dataset_samsum["test"][0]["dialogue"])
    print("\nSummary:")
    print(dataset_samsum["test"][0]["summary"])

    pipe = pipeline("summarization", model=model_ckpt)
    pipe_out = pipe(dataset_samsum["test"][0]["dialogue"])
    print("Summary:")
    print(pipe_out[0]["summary_text"].replace(" .<n>", ".\n"))

    rouge_metric = evaluate.load("rouge")

    score = evaluate_summaries_pegasus(dataset_samsum["test"],
                                       rouge_metric, model,
                                       tokenizer,
                                       column_text="dialogue",
                                       column_summary="summary",
                                       batch_size=8)
    rouge_dict = dict((rn, score[rn].mid.fmeasure) for rn in
                      rouge_names)
    df = pd.DataFrame(rouge_dict, index=["pegasus"])
    print(df)

    dataset_samsum_pt = dataset_samsum.map(convert_examples_to_features, batched=True)
    columns = ["input_ids", "labels", "attention_mask"]
    dataset_samsum_pt.set_format(type="torch", columns=columns)

    # setup data collator
    from transformers import DataCollatorForSeq2Seq, TrainingArguments, Trainer
    seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    train_args = TrainingArguments(output_dir='pegasus-samsum', num_train_epochs=2, warmup_steps=500,
                                   per_device_train_batch_size=1, per_device_eval_batch_size=1, weight_decay=0.01,
                                   logging_steps=10, push_to_hub=False,
                                   eval_strategy="steps", eval_steps=500, save_steps=1e6,
                                   gradient_accumulation_steps=16)
    # gradient_accumulation_steps: 因为模型比较大，batch_size 设置为1，但过小的batch size会导致模型不易收敛，因此采用一种技术叫做：gradient
    # accumulation（梯度累计），正如名字一样，不再一次性计算整个batch的微分，而是通过更新batch，并做微分聚合的方式。当聚合了足够多的微分之后，
    # 运行优化步骤，虽然慢但省GPU内存。
    trainer = Trainer(model=model, args=train_args, tokenizer=tokenizer, data_collator=seq2seq_data_collator,
                      train_dataset=dataset_samsum_pt['train'], eval_dataset=dataset_samsum_pt['test'])
    trainer.train()

    # 评估模型效果
    score = evaluate_summaries_pegasus(dataset_samsum_pt['test'], rouge_metric, trainer.model, tokenizer,
                                       batch_size=2, column_text='dialogue', column_summary='summary')

    rouge_dict = dict((rn, score[rn].mid.fmeasure) for rn in rouge_names)
    df = pd.DataFrame(rouge_dict, index=["pegasus"])
    print(df)
