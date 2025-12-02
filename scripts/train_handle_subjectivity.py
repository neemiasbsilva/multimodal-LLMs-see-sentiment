import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import time
import random
import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm
from scipy import stats
from transformers import pipeline, DistilBertTokenizer, AutoTokenizer, TrainingArguments
from transformers import AutoModelForSequenceClassification, AutoConfig, BartForSequenceClassification, BartTokenizerFast
from utils.other_utils import load_config
from utils.data_loader import data_loader
from scripts.utils_dl import (
    load_experiment_data, log_metrics, save_metrics_to_csv
)
from models.model_handle_subjectivity import (
    DistilBERTModel, ModernBERTModel, Llama3
)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.optim import AdamW
from sklearn.model_selection import KFold, train_test_split
from trl import SFTTrainer
from peft import LoraModel, LoraConfig
from datasets import Dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cuda":
    print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
warnings.filterwarnings("ignore")


# --- Helper Functions for Regression ---

def compute_regression_metrics(preds, targets):
    """Compute MSE, MAE and Pearson Correlation."""
    preds = np.array(preds)
    targets = np.array(targets)
    
    mse = mean_squared_error(targets, preds)
    mae = mean_absolute_error(targets, preds)
    
    # Handle case where correlation is undefined (constant input)
    if len(preds) > 1 and np.std(preds) > 0 and np.std(targets) > 0:
        pearson_corr, _ = stats.pearsonr(preds, targets)
    else:
        pearson_corr = 0.0
        
    return mse, mae, pearson_corr

def train_one_epoch_reg(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    for batch in dataloader:
        ids = batch['ids'].to(device, dtype=torch.long)
        mask = batch['mask'].to(device, dtype=torch.long)
        token_type_id = batch.get('token_type_ids', None)
        if token_type_id is not None:
            token_type_id = token_type_id.to(device, dtype=torch.long)
            
        targets = batch['targets'].to(device, dtype=torch.float)

        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(ids, mask, token_type_id)
        
        # Check if outputs is a Hugging Face object (like BART)
        if hasattr(outputs, "logits"):
            outputs = outputs.logits
        
        # FIX: Use .view(-1) instead of .squeeze()
        # This ensures shape is always [Batch_Size], even if Batch_Size is 1.
        outputs = outputs.view(-1)
        
        loss = loss_fn(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        all_preds.extend(outputs.detach().cpu().numpy())
        all_targets.extend(targets.detach().cpu().numpy())
        
    return total_loss / len(dataloader), all_preds, all_targets

def validate_one_epoch_reg(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            ids = batch['ids'].to(device, dtype=torch.long)
            mask = batch['mask'].to(device, dtype=torch.long)
            token_type_id = batch.get('token_type_ids', None)
            if token_type_id is not None:
                token_type_id = token_type_id.to(device, dtype=torch.long)
                
            targets = batch['targets'].to(device, dtype=torch.float)

            # Forward pass
            outputs = model(ids, mask, token_type_id)
            
            # Check if outputs is a Hugging Face object
            if hasattr(outputs, "logits"):
                outputs = outputs.logits

            # FIX: Use .view(-1) instead of .squeeze()
            outputs = outputs.view(-1)
            
            loss = loss_fn(outputs, targets)
            total_loss += loss.item()
            
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
    return total_loss / len(dataloader), all_preds, all_targets


def val_reg(log_dir, model, dataloader, loss_fn, kfold, df_metrics, start_time):
    """Validation wrapper for regression."""
    val_loss, preds, targets = validate_one_epoch_reg(model, dataloader, loss_fn, device)
    mse, mae, pearson = compute_regression_metrics(preds, targets)
    
    # Update metrics DF
    new_row = pd.DataFrame({
        "kfold": [kfold + 1],
        "val_mse": [mse],
        "val_mae": [mae],
        "val_pearson": [pearson],
        "time_sec": [int(time.time() - start_time)]
    })
    df_metrics = pd.concat([df_metrics, new_row], axis=0)
    save_metrics_to_csv(df_metrics, log_dir)
    
    return df_metrics, preds, targets, pearson

# --- Llama Specifics for Regression (Prompting for Score) ---

def train_llama_qlora_reg(model, tokenizer, train_data, eval_data, log_dir, epochs, batch_size, max_len):
    peft_config = LoraConfig(
        lora_alpha=8, lora_dropout=0.1, r=32, bias="none", task_type="CAUSAL_LM"
    )
    training_arguments = TrainingArguments(
        output_dir=log_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=1,
        optim="paged_adamw_32bit",
        logging_steps=25,
        learning_rate=2e-4,
        fp16=True,
        save_strategy="no",
        report_to="tensorboard",
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=eval_data,
        peft_config=peft_config,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=training_arguments,
        packing=False,
        max_seq_length=max_len,
    )
    trainer.train()
    trainer.save_model(f"{log_dir}/results/trained_model")
    return model, tokenizer

def predict_llama_reg(X_test, model, tokenizer):
    """Predict scores for test data using Llama."""
    y_pred = []
    pipe = pipeline(
        task="text-generation", model=model, tokenizer=tokenizer, max_new_tokens=5, temperature=0.01 # Low temp for deterministic numbers
    )
    for i in tqdm(range(len(X_test))):
        prompt = X_test.iloc[i]["text"]
        # Basic inference logic
        result = pipe(prompt)
        # Assuming prompt ends with "Score: "
        generated_text = result[0]['generated_text']
        
        # Logic to extract the number after the last "=" or ":"
        try:
            answer = generated_text.split("=")[-1].strip()
            score = float(answer)
        except ValueError:
            # Fallback cleanup
            try:
                import re
                # Find the last float number in the string
                matches = re.findall(r"[-+]?\d*\.\d+|\d+", answer)
                score = float(matches[0]) if matches else 0.0 # Default to 0.0 if fail
            except:
                score = 0.0 
                
        y_pred.append(score)
    return y_pred

# --- Main Fit Function ---

def fit_reg(
    model, epochs, optimizer,
    train_dl, val_dl,
    log_dir, checkpoint_dir,
    fold, device
):
    # MSE Loss for Regression
    loss_fn = torch.nn.MSELoss() 
    
    # Freeze backbone logic (Keep same as original if needed)
    # ... (omitted for brevity, assume backbone freezing logic is same)

    log_file = os.path.join(log_dir, f"training_logs_{fold+1:02d}.txt")
    open(log_file, 'w').close()

    df_metrics = pd.DataFrame([])
    best_pearson = -1.0 
    patience = 10
    patience_counter = 0

    for epoch in tqdm(range(epochs)):
        # Train
        train_loss, _, _ = train_one_epoch_reg(model, train_dl, optimizer, loss_fn, device)
        
        # Validation
        val_loss, preds_val, targets_val = validate_one_epoch_reg(model, val_dl, loss_fn, device)
        mse_val, mae_val, pearson_val = compute_regression_metrics(preds_val, targets_val)

        # Logging
        log_msg = f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | MAE: {mae_val:.4f} | Pearson: {pearson_val:.4f}"
        print(log_msg)
        with open(log_file, "a") as f:
            f.write(log_msg + "\n")

        # Save Checkpoint based on Pearson Correlation (or MSE)
        if pearson_val > best_pearson:
            best_pearson = pearson_val
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"best_model_fold{fold}.pt"))
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

        # Update DataFrame
        df_metrics = pd.concat([df_metrics, pd.DataFrame({
            "epoch": [epoch + 1],
            "train_loss": [train_loss],
            "val_loss": [val_loss],
            "val_mae": [mae_val],
            "val_pearson": [pearson_val],
        })], axis=0)
        df_metrics.to_csv(os.path.join(log_dir, f"training_logs_{fold+1:02d}.csv"), index=False)

    return model, loss_fn

# --- Main Train Function ---

def train(config, config_path):
    print(f'Train Regression Experiment: {config["experiment_name"]}')

    learning_rate = float(config["learning_rate"])
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    max_len = config["max_len"]
    model_path = config["model_path"]
    log_dir = config["log_dir"]
    checkpoint_dir = config["checkpoint_dir"]
    model_name = config["model_name"]
    
    alpha_version = int(config_path.split('/')[-2].split('-')[-1][-1])
    experiment_group = config_path.split('/')[-2].split('-')[-2]
    if config_path.split('/')[-2].split('-')[0] == "openai":
        dataset_type = "gpt4-openai-classify" 
        if config_path.split('/')[-3] == "experiments-twitter":
            flag_twiter = True
            print(f"Twitter dataset")
    elif config_path.split('/')[-2].split('-')[0] == "deepseek":
        dataset_type = "deepseek"
    elif config_path.split('/')[-2].split('-')[0] == "gemini":
        dataset_type = "gemini"
    else:
        dataset_type = "minigpt4-classify"

    # Load Data
    df = load_experiment_data(alpha_version, dataset_type, experiment_group)
    if experiment_group == "p3":
        mapping = {
            2:1,
            0:0,
            1:-1,
        }
        df["sentiment"] = df["sentiment"].map(mapping)
    print(df)
    # CRITICAL: Ensure targets are floats for regression
    df["sentiment"] = df["sentiment"].astype(float)
    
    # Only using train/val split from kfold, assuming no separate test set per original logic
    train_val_df = df.copy()
    
    train_params = {"batch_size": batch_size, "shuffle": True}
    val_params = {"batch_size": batch_size, "shuffle": False}

    if model_name in ["distil-bert", "modern-bert", "bart"]:
        if model_name == "distil-bert":
            tokenizer = DistilBertTokenizer.from_pretrained(model_path, do_lower_case=True)
        elif model_name == "modern-bert":
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        elif model_name == "bart":
            tokenizer = BartTokenizerFast.from_pretrained(model_path)

        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        df_fold_results = pd.DataFrame([])

        for fold, (train_idx, val_idx) in enumerate(kfold.split(train_val_df)):
            print(f"Fold {fold + 1}")
            start_time = time.time()
            
            train_df = train_val_df.iloc[train_idx].copy().reset_index(drop=True)
            val_df = train_val_df.iloc[val_idx].copy().reset_index(drop=True)

            # Data Loaders
            train_dl = data_loader(train_df, tokenizer, max_len, train_params)
            val_dl = data_loader(val_df, tokenizer, max_len, val_params)

            # Model Init
            if model_name == "distil-bert":
                model = DistilBERTModel(model_path) # No class_size arg needed, hardcoded to 1
            elif model_name == "modern-bert":
                model = ModernBERTModel(model_path)
            elif model_name == "bart":
                # Bart supports regression natively with num_labels=1
                model = BartForSequenceClassification.from_pretrained(
                    model_path,
                    num_labels=1,              # Set to 1 for regression
                    problem_type="regression", # Explicitly set problem type
                    ignore_mismatched_sizes=True # <--- ADD THIS LINE
                )
                # model = BartForSequenceClassification.from_pretrained(
                #     model_path, num_labels=1, problem_type="regression"
                # )

            model.to(device)
            optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-6)

            # Train Loop
            model, loss_fn = fit_reg(
                model, epochs, optimizer,
                train_dl, val_dl,
                log_dir, checkpoint_dir,
                fold, device
            )

            # Final Eval
            df_fold_results, y_pred, y_true, pearson = val_reg(log_dir, model, val_dl, loss_fn, fold, df_fold_results, start_time)
            
            # Save predictions
            result_df = pd.DataFrame({
                "id": val_df["id"].to_list(),
                "text": val_df["text"].to_list(),
                "target_score": y_true,
                "predicted_score": y_pred
            })
            result_df.to_csv(os.path.join(log_dir, f"test_logs_{fold+1:02d}.csv"), index=False)
            
            del model
            torch.cuda.empty_cache()

        print(f"Mean Pearson Correlation: {np.mean(df_fold_results['val_pearson']):.4f}")

    elif model_name == "llama-qlora":
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        df_metrics = pd.DataFrame([])

        for fold, (train_idx, val_idx) in enumerate(kfold.split(train_val_df)):
            print(f"Fold {fold + 1}")
            start_time = time.time()
            
            llama3 = Llama3(model_path)
            model, tokenizer = llama3.get_model()
            
            train_df = train_val_df.iloc[train_idx].copy()
            val_df = train_val_df.iloc[val_idx].copy()

            # Prompt Engineering for Regression
            # asking for a score specifically
            prompt_template = """Analyze the sentiment of the following text and assign a score between 0.0 and 1.0 (where 0 is most negative and 1 is most positive).\nText: """
            
            train_df["text"] = train_df.apply(lambda x: f"{prompt_template}{x['text']}\nScore={x['sentiment']}", axis=1)
            # For validation, we leave the score empty to generate it
            val_df["text_prompt"] = val_df.apply(lambda x: f"{prompt_template}{x['text']}\nScore=", axis=1) 
            # keep original text column for internal Trainer usage if needed, but we need the prompted version
            
            # Convert to HF Dataset
            train_data = Dataset.from_pandas(train_df[["text"]])
            eval_data = Dataset.from_pandas(val_df.rename(columns={"text": "original_text", "text_prompt": "text"})[["text"]])

            # Train
            model, tokenizer = train_llama_qlora_reg(
                model, tokenizer, train_data, eval_data, log_dir, epochs, batch_size, max_len
            )

            # Inference
            y_pred = predict_llama_reg(val_df.rename(columns={"text_prompt": "text"}), model, tokenizer)
            y_true = val_df["sentiment"].tolist()

            mse, mae, pearson = compute_regression_metrics(y_pred, y_true)
            
            df_metrics = pd.concat([df_metrics, pd.DataFrame({
                "kfold": [fold + 1], "mse": [mse], "mae": [mae], "pearson": [pearson]
            })])
            df_metrics.to_csv(os.path.join(log_dir, "test_logs.csv"), index=False)
            
            llama3.cleanup()
            
        print(f"Mean Pearson: {np.mean(df_metrics['pearson']):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the DL model (Regression).")
    parser.add_argument(
        "--config", type=str, default="experiments/experiment1/config.yaml"
    )
    args = parser.parse_args()
    config_path = args.config
    config = load_config(config_path)
    train(config, config_path)