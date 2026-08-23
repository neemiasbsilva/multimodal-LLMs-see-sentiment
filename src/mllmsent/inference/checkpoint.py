import argparse
import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm
from transformers import (
    pipeline, DistilBertTokenizer, AutoTokenizer, 
    BartForSequenceClassification, BartTokenizerFast
)
from mllmsent.config import load_yaml as load_config
from mllmsent.data.loader import data_loader
from mllmsent.models.heads import (
    DistilBERTModel, ModernBERTModel, Llama3
)
from peft import PeftModel
import gc

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Use GPU 1 (index 1)

device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cuda":
    print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
warnings.filterwarnings("ignore")


def load_bart_model(checkpoint_path, model_path, num_classes):
    """Load BART model from checkpoint."""
    print(f"Loading BART model from checkpoint: {checkpoint_path}")
    
    # Initialize the BART model
    model = BartForSequenceClassification.from_pretrained(
        model_path,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )
    
    # Load the trained weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    
    # Load tokenizer
    tokenizer = BartTokenizerFast.from_pretrained(model_path)
    
    return model, tokenizer


def load_modernbert_model(checkpoint_path, model_path, num_classes):
    """Load ModernBERT model from checkpoint."""
    print(f"Loading ModernBERT model from checkpoint: {checkpoint_path}")
    
    # Initialize the ModernBERT model
    model = ModernBERTModel(model_path, num_classes)
    
    # Load the trained weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    return model, tokenizer


def load_llama_model(checkpoint_dir, model_path):
    """Load Llama model from checkpoint directory."""
    print(f"Loading Llama model from checkpoint directory: {checkpoint_dir}")
    
    # Initialize Llama model
    llama3 = Llama3(model_path)
    model, tokenizer = llama3.get_model()
    
    # Load the trained LoRA weights
    model = PeftModel.from_pretrained(model, checkpoint_dir)
    model.to(device)
    model.eval()
    
    return model, tokenizer


def predict_bart_modernbert(model, tokenizer, texts, batch_size=32, max_len=512):
    """Predict using BART or ModernBERT models."""
    predictions = []
    
    # Create dataset
    df = pd.DataFrame({"text": texts})
    params = {"batch_size": batch_size, "shuffle": False}
    dataloader = data_loader(df, tokenizer, max_len, params)
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            ids = batch["ids"].to(device)
            mask = batch["mask"].to(device)
            token_type_id = batch["token_type_id"].to(device)
            
            outputs = model(ids, mask, token_type_id)
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
                
            preds = torch.argmax(logits, dim=1)
            predictions.extend(preds.cpu().numpy())
    
    return predictions


def predict_llama(model, tokenizer, texts, num_classes, experiment_group):
    """Predict using Llama model."""
    predictions = []
    
    # Define prompts based on number of classes and experiment group
    if num_classes == 2 and experiment_group == "p2plus":
        prompt = """What is the sentiment of this description? Please choose an answer from \
            {"Negative": 1, "Neutral": 0}
        """
    elif num_classes == 2 and experiment_group == "p2neg":
        prompt = """What is the sentiment of this description? Please choose an answer from \
            {"Positive": 1, "Neutral": 0}
        """
    elif num_classes == 3:
        prompt = """What is the sentiment of this description? Please choose an answer from \
            {"Positive": 2, "Negative": 0, "Neutral": 1}
        """
    else:
        prompt = """What is the sentiment of this description? Please choose an answer from \
            {"Positive": 4, "SlightlyPositive": 3, "Neutral": 2, "SlightlyNegative": 1, "Negative": 0}
        """
    
    # Create pipeline for text generation
    pipe = pipeline(
        task="text-generation", 
        model=model, 
        tokenizer=tokenizer, 
        max_new_tokens=1, 
        temperature=0.01, 
        do_sample=True
    )
    
    for text in tqdm(texts, desc="Predicting with Llama"):
        full_prompt = f"{prompt}{text}="
        result = pipe(full_prompt)
        answer = result[0]['generated_text'].split("=")[-1].strip().strip(".,!?;'\"")
        
        if answer.isdigit():
            predictions.append(int(answer))
        else:
            # Handle invalid predictions by using a random class
            import random
            predictions.append(random.randint(0, num_classes - 1))
    
    return predictions


def inference(
    model_name, 
    checkpoint_path, 
    model_path, 
    texts, 
    num_classes, 
    experiment_group=None,
    batch_size=32, 
    max_len=512
):
    """
    Perform inference using the specified model.
    
    Args:
        model_name (str): Name of the model ("bart", "modern-bert", "llama")
        checkpoint_path (str): Path to the model checkpoint
        model_path (str): Path to the base model
        texts (list): List of texts to predict on
        num_classes (int): Number of classes for classification
        experiment_group (str): Experiment group for Llama prompts
        batch_size (int): Batch size for inference
        max_len (int): Maximum sequence length
    
    Returns:
        list: Predictions for the input texts
    """
    
    if model_name == "bart":
        model, tokenizer = load_bart_model(checkpoint_path, model_path, num_classes)
        predictions = predict_bart_modernbert(model, tokenizer, texts, batch_size, max_len)
        
    elif model_name == "modern-bert":
        model, tokenizer = load_modernbert_model(checkpoint_path, model_path, num_classes)
        predictions = predict_bart_modernbert(model, tokenizer, texts, batch_size, max_len)
        
    elif model_name == "llama":
        if experiment_group is None:
            raise ValueError("experiment_group is required for Llama inference")
        model, tokenizer = load_llama_model(checkpoint_path, model_path)
        predictions = predict_llama(model, tokenizer, texts, num_classes, experiment_group)
        
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    
    return predictions


def main():
    parser = argparse.ArgumentParser(description="Inference script for trained models.")
    parser.add_argument("--model_name", type=str, required=True, 
                       choices=["bart", "modern-bert", "llama"],
                       help="Name of the model to use for inference")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                       help="Path to the model checkpoint")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the base model")
    parser.add_argument("--input_file", type=str, required=True,
                       help="Path to CSV file containing texts to predict on")
    parser.add_argument("--output_file", type=str, required=True,
                       help="Path to save predictions")
    parser.add_argument("--num_classes", type=int, required=True,
                       help="Number of classes for classification")
    parser.add_argument("--experiment_group", type=str, default=None,
                       help="Experiment group (required for Llama)")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for inference")
    parser.add_argument("--max_len", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--text_column", type=str, default="text",
                       help="Name of the text column in the input CSV")
    
    args = parser.parse_args()
    
    # Load input data
    print(f"Loading input data from: {args.input_file}")
    df = pd.read_csv(args.input_file)
    
    if args.text_column not in df.columns:
        raise ValueError(f"Column '{args.text_column}' not found in the input file")
    
    texts = df[args.text_column].tolist()
    print(f"Loaded {len(texts)} texts for inference")
    
    # Perform inference
    print(f"Starting inference with {args.model_name} model...")
    predictions = inference(
        model_name=args.model_name,
        checkpoint_path=args.checkpoint_path,
        model_path=args.model_path,
        texts=texts,
        num_classes=args.num_classes,
        experiment_group=args.experiment_group,
        batch_size=args.batch_size,
        max_len=args.max_len
    )
    
    # Save results
    results_df = df.copy()
    results_df['prediction'] = predictions
    results_df.to_csv(args.output_file, index=False)
    
    print(f"Predictions saved to: {args.output_file}")
    print(f"Prediction distribution: {pd.Series(predictions).value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    main() 