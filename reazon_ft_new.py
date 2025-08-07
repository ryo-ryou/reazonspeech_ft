import os
import torch
import numpy as np
from datasets import load_dataset, DatasetDict, Audio
import evaluate
import wandb
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)

# ==== 1. Wandb のログディレクトリ設定・初期化 ====
wandb_dir = "/workspace/Dounloads/dysarthria-speech/src/wandb_tmp"
os.environ["WANDB_DIR"] = wandb_dir
os.environ["WANDB_CACHE_DIR"] = os.path.join(wandb_dir, "cache")
os.environ["WANDB_CONFIG_DIR"] = os.path.join(wandb_dir, "config")
wandb.init(project="whisper-finetune-large-v3")

# ==== 2. モデル名・データセット設定 ====
model_name = "openai/whisper-large-v3"
dataset_name = "reazon-research/reazonspeech"
dataset_config = "tiny"  # 適宜

# ==== 3. Whisper Processor・モデル準備 ====
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)
model.config.forced_decoder_ids = None  # 明示しておく
model.config.use_cache = False

# ==== 4. データセット読み込み====

dataset = load_dataset(dataset_name, dataset_config)

# validation 分割なければ分割
if "validation" not in dataset:
    if "train" in dataset:
        train_valid = dataset["train"].train_test_split(test_size=0.1)
        dataset = DatasetDict({"train": train_valid["train"], "validation": train_valid["test"]})
    else:
        dataset = dataset.train_test_split(test_size=0.1)
        dataset["validation"] = dataset.pop("test")
print("Dataset splits:", dataset.keys())


# ==== 5. データ整形 ====（必ず各カラム名合ってるかチェック）
def preprocess_function(batch):
    audio = batch["audio"]
    input_features = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    # transcriptionカラムはご自身のデータに合わせて変更
    labels = processor.tokenizer(
        batch["transcription"],
        max_length=448, padding="max_length", truncation=True, return_tensors="pt",
    ).input_ids[0]
    batch["input_features"] = input_features
    batch["labels"] = labels
    return batch

for split in dataset.keys():
    dataset[split] = dataset[split].map(
        preprocess_function,
        remove_columns=dataset[split].column_names,
        num_proc=1,
    )

# ==== 6. DataCollator ====
from dataclasses import dataclass
from typing import List, Dict, Union, Optional

@dataclass
class DataCollatorWhisper:
    processor: WhisperProcessor
    padding: Union[bool, str] = True
    return_tensors: Optional[str] = "pt"

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        input_feat = [{"input_features": f["input_features"]} for f in features]
        label_feat = [{"input_ids": f["labels"]} for f in features]
        batch = self.processor.feature_extractor.pad(
            input_feat, padding=self.padding, return_tensors=self.return_tensors
        )
        labels_batch = self.processor.tokenizer.pad(
            label_feat, padding=self.padding, return_tensors=self.return_tensors
        )
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch

data_collator = DataCollatorWhisper(processor=processor, return_tensors="pt", padding=True)

# ==== 7. メトリクス関数（型崩れ考慮）====
cer_metric = evaluate.load("cer")

def to_numpy(arr):
    if hasattr(arr, "cpu"):
        arr = arr.cpu()
    if hasattr(arr, "numpy"):
        arr = arr.numpy()
    return arr

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    
    import torch
    
    # logitsがnumpyの場合tensorに変換
    if not isinstance(logits, torch.Tensor):
        logits = torch.tensor(logits)
    
    # 予測IDをバッチ単位で正しく取得
    # shape: (batch_size, seq_len) になるように調整
    if logits.dim() == 1:
        pred_ids = logits.unsqueeze(0).argmax(dim=-1)
    else:
        pred_ids = logits.argmax(dim=-1)
    
    # tokenizer.batch_decodeは、トークンIDのリストのリストを渡す必要あり
    # pred_idsが1次元のTensorの場合は2次元にする
    if pred_ids.dim() == 1:
        pred_ids = pred_ids.unsqueeze(0)
    
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    
    # labelsをnumpy配列に変換して -100 をpad_token_idに置換
    def to_numpy(arr):
        if hasattr(arr, "cpu"):
            arr = arr.cpu()
        if hasattr(arr, "numpy"):
            arr = arr.numpy()
        return arr

    labels = to_numpy(labels)
    labels[labels == -100] = processor.tokenizer.pad_token_id
    
    label_str = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    return {"cer": cer}


# ==== 8. トレーニング引数 ====
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    eval_strategy="steps",
    save_strategy="steps",
    save_steps=100,
    eval_steps=100,
    logging_steps=20,
    num_train_epochs=3,
    warmup_steps=150,
    learning_rate=1e-6,
    predict_with_generate=True,
    load_best_model_at_end=True,
    metric_for_best_model="cer",
    greater_is_better=False,
    save_total_limit=5,
    report_to=["wandb"],
    run_name="whisper-large-v3-finetune-reazonspeech",
    push_to_hub=False,
)

# ==== 9. トレーナーインスタンス化（tokenizerをprocessorやfeature_extractorではなくtokenizerにするのがよい） ====
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=data_collator,
    tokenizer=processor.tokenizer,
    compute_metrics=compute_metrics,
)

# ==== 10. トレーニング ====
trainer.train()

wandb.finish()
