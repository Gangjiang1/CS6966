# -*- coding: utf-8 -*-
"""CS6966_HW1 BY GANG JIANG
"""

from datasets import load_dataset, load_metric
import numpy as np
dataset = load_dataset('imdb')
metric = load_metric('accuracy')

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base", use_fast=True)


task_to_keys = {
    "imdb": ("text", None),
}

sentence1_key, sentence2_key = task_to_keys["imdb"]

def preprocess_function(examples):
    if sentence2_key is None:
        return tokenizer(examples[sentence1_key], truncation=True)
    return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True)

encoded_dataset = dataset.map(preprocess_function, batched=True)

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

num_labels = 2
model = AutoModelForSequenceClassification.from_pretrained('microsoft/deberta-v3-base', num_labels=num_labels)

metric_name = "accuracy"
model_name = "microsoft/deberta-v3-base".split("/")[-1]
task="imdb"
batch_size = 4
args = TrainingArguments(
    f"{model_name}-finetuned-{task}",
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=1,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)
# small_train_dataset=encoded_dataset['train'].shuffle(seed=42).select(range(100))
# small_test_dataset=encoded_dataset['test'].shuffle(seed=42).select(range(100))

trainer = Trainer(
    model,
    args,
    train_dataset=encoded_dataset['train'],
    eval_dataset=encoded_dataset['test'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

results = trainer.evaluate()

print("Evaluation results:", results['eval_accuracy'])

import jsonlines
import random

# Assuming you have already trained the model and obtained predictions
predictions = trainer.predict(encoded_dataset['test'])

# Get the predicted labels and true labels
predicted_labels = np.argmax(predictions.predictions, axis=1)
true_labels = encoded_dataset['test']['label']

# Collect incorrect predictions and their information
incorrectly_predicted = []
for i in range(len(predicted_labels)):
    if predicted_labels[i] != true_labels[i]:
        review_text = encoded_dataset['test']['text'][i]
        gold_label = encoded_dataset['test']['label'][i]
        predicted_label = int(predicted_labels[i])
        incorrectly_predicted.append({
            'review': review_text,
            'label': gold_label,
            'predicted': predicted_label,
        })

# Randomly sample 10 instances
random_sample = random.sample(incorrectly_predicted, min(10, len(incorrectly_predicted)))

# Save the sampled instances in a JSONLines file
with jsonlines.open('incorrect_predictions.jsonl', 'w') as writer:
    for item in random_sample:
        writer.write(item)


print("Saved 10 randomly sampled incorrectly predicted instances to jsonl.")
