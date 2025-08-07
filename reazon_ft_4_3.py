import os
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import sys
import transformers

import torch
from datasets import load_from_disk
from evaluate import load
from transformers import (
    WhisperForConditionalGeneration,
    WhisperFeatureExtractor,
    WhisperProcessor,
    WhisperTokenizer,
    TrainingArguments,
    Trainer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
import inspect # inspectモジュールをインポート


print("--- 詳細診断情報 ---")
print(f"Pythonのパス: {sys.executable}")
print(f"Transformersのバージョン: {transformers.__version__}")
# ライブラリ本体の場所を確認
print(f"Transformersライブラリのパス: {transformers.__file__}")
# TrainingArgumentsクラスがどこから来たかを確認
print(f"TrainingArgumentsクラスのソース: {TrainingArguments.__module__}")
print("--------------------")

# --- 1. 評価指標を計算する関数 ---
# WER (単語誤り率) と CER (文字誤り率) のメトリクスをロード
wer_metric = load("wer")
cer_metric = load("cer")

# グローバルスコープにtokenizerを配置して、関数内からアクセスできるようにする
tokenizer = None

def compute_metrics(pred):
    # pred.predictions が logits の場合があるので、生成したトークン列は pred.predictions と別に pred.predictions に入る可能性あり
    if pred.predictions is None:
        return {"wer": 1.0, "cer": 1.0}

    # pred.predictions は int のトークンID列の2次元配列（バッチ×長さ）が入る想定
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # -100 を pad_token_id に置換
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer, "cer": cer}


# --- 2. 音声とテキストのバッチを作成するカスタムデータコレーター ---
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # 音声特徴量をパディング
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # テキストラベルをパディング
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # ラベルのパディング部分を-100に置換 (損失計算で無視するため)
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels
        return batch

# --- 3. 設定と準備 ---
# 各種パス（ご自身の環境に合わせて変更してください）
preprocessed_dataset_path = "/workspace/Dounloads/dysarthria-speech/src/preprocessed_dataset/pre_data"
output_dir = "/workspace/Dounloads/dysarthria-speech/src/whisper-finetuned4"

# モデル名
model_name = "openai/whisper-large-v3"

# 前処理済みデータセットをロード
common_voice = load_from_disk(preprocessed_dataset_path)

# Whisperの各コンポーネントをロード
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
# compute_metrics関数で使うため、グローバル変数のtokenizerを初期化
tokenizer = WhisperTokenizer.from_pretrained(model_name, language="japanese", task="transcribe")
processor = WhisperProcessor.from_pretrained(model_name, language="japanese", task="transcribe")

# モデルをロード
model = WhisperForConditionalGeneration.from_pretrained(model_name)
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

# 30秒を超える音声をチャンクに分割するための前処理関数
def chunk_long_audio(batch):
    # バッチから音声データとテキストを取り出す
    # ReazonSpeechの正式な列名 'audio' と 'transcription' を使用
    audio_list = batch["audio"]
    transcription_list = batch["transcription"] # "sentence" から "transcription" に修正
    
    new_samples = {"input_features": [], "labels": []}
    
    for i in range(len(audio_list)):
        audio_data = audio_list[i]
        text_data = transcription_list[i]
        
        # 音声データをメルスペクトログラムに変換
        # audio_dataは辞書{'array': ..., 'sampling_rate': ...}になっている
        inputs = feature_extractor(audio_data["array"], sampling_rate=audio_data["sampling_rate"], return_tensors="pt")
        
        # テキストをトークンIDに変換
        labels = tokenizer(text_data).input_ids

        # メルスペクトログラムの時間次元の長さを取得 (Whisperは3000フレーム=30秒)
        max_length = 3000
        input_length = inputs.input_features.shape[2]

        # 30秒以下ならそのまま追加
        if input_length <= max_length:
            new_samples["input_features"].append(inputs.input_features.squeeze(0))
            new_samples["labels"].append(labels)
        # 30秒を超える場合はチャンクに分割
        else:
            for start in range(0, input_length, max_length):
                end = start + max_length
                chunk = inputs.input_features[:, :, start:end]
                # パディングを追加して必ず3000フレームにする
                padded_chunk = torch.nn.functional.pad(chunk, (0, max_length - chunk.shape[2]))
                
                new_samples["input_features"].append(padded_chunk.squeeze(0))
                # 各音声チャンクに、元の書き起こし全文をラベルとして割り当てる
                new_samples["labels"].append(labels)

    return new_samples

# 作成した関数をデータセット全体に適用
# `batched=True`で処理を高速化し、不要な列を削除する
print("音声データのチャンク化を開始します...")
column_names = common_voice["train"].column_names
common_voice = common_voice.map(chunk_long_audio, batched=True, remove_columns=column_names)
print("音声データのチャンク化が完了しました。")
print("変換後のデータセットのサンプル数（学習）:", len(common_voice["train"]))

# WandB設定 (オプション)
os.environ["WANDB_PROJECT"] = "whisper-finetuning-project"
os.environ["WANDB_NAME"] = "reazonspeech-large-v3-cer-top5"

"""
print("\n--- TrainingArguments クラスの最終診断 ---")
print(inspect.signature(transformers.TrainingArguments.__init__))

try:
    # TrainingArgumentsクラスが定義されているファイルパスを取得
    file_path = inspect.getsourcefile(TrainingArguments)
    print(f"TrainingArgumentsのファイルパス: {file_path}")

    # __init__メソッドのソースコードを取得して、'evaluation_strategy'があるか確認
    init_source = inspect.getsource(TrainingArguments.__init__)
    if "evaluation_strategy" in init_source:
        print("OK: __init__メソッドに 'evaluation_strategy' 引数が存在します。")
    else:
        print("エラー: __init__メソッドに 'evaluation_strategy' 引数がありません！ (古いバージョン)")
except Exception as e:
    print(f"inspectでエラーが発生しました: {e}")
print("---------------------------------------\n")
"""

# --- 4. トレーニング設定 ---
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,       # ← 評価バッチは小さくする
    eval_accumulation_steps=16,         # ← 評価時にテンソルを分割して保持
    gradient_accumulation_steps=4,      # メモリ節約のため1推奨
    learning_rate=1e-7,
    warmup_steps=500,
    num_train_epochs=1,
    fp16=True,
    #gradient_checkpointing=True,
    eval_strategy="steps",
    eval_steps=50,#5000
    save_strategy="steps",
    save_steps=50,#5000
    logging_steps=25,#200
    load_best_model_at_end=True,
    metric_for_best_model="cer",     # 評価基準を'cer'に設定
    save_total_limit=5,              # 保存するチェックポイントの最大数を5に設定
    greater_is_better=False,         # CERは低い方が良いためFalse
    predict_with_generate=True,
    report_to="wandb",
    dataloader_num_workers=2,
    gradient_checkpointing=False,

)

# カスタムデータコレーターをインスタンス化
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
#model.config.use_cache = False   # 明示的に無効化

# --- 5. Trainerの作成と学習の実行 ---
#trainer = Trainer(

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=common_voice["train"],
    #eval_dataset=common_voice["validation"],
    eval_dataset = common_voice["validation"].select(range(250)),
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    #tokenizer=processor.feature_extractor,
    tokenizer=processor,
    #predict_with_generate=True, 
)

# ファインチューニングを開始
trainer.train()

# 学習結果（トップ5のうち最も良かったモデル）を保存
trainer.save_model(os.path.join(output_dir, "best_model"))
processor.save_pretrained(os.path.join(output_dir, "best_model"))

# テストデータで最終評価
print("\n--- Test Set Evaluation with the Best Model ---")
predictions = trainer.predict(common_voice["test"])
print(predictions.metrics)