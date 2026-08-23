# Inference Scripts Documentation

This directory contains scripts for running inference with trained models (BART, ModernBERT, and Llama) for sentiment analysis.

## Files Overview

- `inference.py` - Main inference script that can load trained models and perform predictions
- `run_inference.py` - Helper script that automatically determines parameters from checkpoint names
- `example_inference.py` - Example script demonstrating different usage patterns

## Quick Start

### 1. Using the Helper Script (Recommended)

The easiest way to run inference is using the `run_inference.py` helper script:

```bash
# List all available checkpoints
python scripts/run_inference.py --list

# Run inference with automatic parameter detection
python scripts/run_inference.py \
    --checkpoint checkpoints/best_checkpoint_gpt4-openai-classify_bart_p5_sigma3_finetuned.pt \
    --input your_data.csv \
    --output predictions.csv
```

### 2. Using the Main Inference Script Directly

For more control, you can use the main inference script directly:

```bash
python scripts/inference.py \
    --model_name bart \
    --checkpoint_path checkpoints/best_checkpoint_gpt4-openai-classify_bart_p5_sigma3_finetuned.pt \
    --model_path facebook/bart-large-mnli \
    --input_file your_data.csv \
    --output_file predictions.csv \
    --num_classes 5 \
    --batch_size 32 \
    --max_len 512
```

## Input Data Format

The input CSV file should contain a column with the texts to classify. By default, the script looks for a column named "text", but you can specify a different column name using the `--text_column` parameter.

Example input CSV:
```csv
text
"This is a great product!"
"The service was terrible."
"I'm neutral about this."
```

## Output Format

The script will create a new CSV file with the original data plus a new "prediction" column containing the predicted class labels.

Example output CSV:
```csv
text,prediction
"This is a great product!",4
"The service was terrible.",0
"I'm neutral about this.",2
```

## Model-Specific Information

### BART Model
- **Checkpoint Format**: `best_checkpoint_{dataset_type}_bart_{experiment_group}_sigma{alpha_version}_{finetuning}.pt`
- **Base Model**: `facebook/bart-large-mnli`
- **Classes**: 
  - p5: 5 classes (0=Negative, 1=SlightlyNegative, 2=Neutral, 3=SlightlyPositive, 4=Positive)
  - p3: 3 classes (0=Negative, 1=Neutral, 2=Positive)
  - p2plus: 2 classes (0=Neutral, 1=Negative)
  - p2neg: 2 classes (0=Neutral, 1=Positive)

### ModernBERT Model
- **Checkpoint Format**: `best_checkpoint_{dataset_type}_modernbert_{experiment_group}_sigma{alpha_version}_{finetuning}.pt`
- **Base Model**: Update with actual ModernBERT path in `run_inference.py`
- **Classes**: Same as BART

### Llama Model
- **Checkpoint Directory**: `checkpoints/llama/{dataset_type}_{experiment_group}_sigma{alpha_version}/`
- **Base Model**: Update with actual Llama path in `run_inference.py`
- **Classes**: Same as BART
- **Note**: Requires `experiment_group` parameter for prompt generation

## Examples

### Example 1: BART Inference
```bash
python scripts/run_inference.py \
    --checkpoint checkpoints/best_checkpoint_gpt4-openai-classify_bart_p5_sigma3_finetuned.pt \
    --input sample_data.csv \
    --output bart_predictions.csv
```

### Example 2: ModernBERT Inference
```bash
python scripts/run_inference.py \
    --checkpoint checkpoints/best_checkpoint_gpt4-openai-classify_modernbert_p3_sigma5_finetuned.pt \
    --input sample_data.csv \
    --output modernbert_predictions.csv
```

### Example 3: Llama Inference (Direct)
```bash
python scripts/inference.py \
    --model_name llama \
    --checkpoint_path checkpoints/llama/gpt4-openai-classify_p2plus_sigma3/ \
    --model_path meta-llama/Llama-2-7b-chat-hf \
    --input_file sample_data.csv \
    --output_file llama_predictions.csv \
    --num_classes 2 \
    --experiment_group p2plus
```

### Example 4: Custom Column Name
```bash
python scripts/inference.py \
    --model_name bart \
    --checkpoint_path checkpoints/best_checkpoint_gpt4-openai-classify_bart_p5_sigma3_finetuned.pt \
    --model_path facebook/bart-large-mnli \
    --input_file your_data.csv \
    --output_file predictions.csv \
    --num_classes 5 \
    --text_column "description"
```

## Running Examples

To see the inference scripts in action, run the example script:

```bash
python scripts/example_inference.py
```

This will:
1. Create sample data
2. Run inference with different models
3. Display results
4. Save output files

## Parameters

### Common Parameters
- `--model_name`: Model type ("bart", "modern-bert", "llama")
- `--checkpoint_path`: Path to model checkpoint
- `--model_path`: Path to base model
- `--input_file`: Input CSV file
- `--output_file`: Output CSV file
- `--num_classes`: Number of classification classes
- `--batch_size`: Batch size for inference (default: 32)
- `--max_len`: Maximum sequence length (default: 512)
- `--text_column`: Name of text column in CSV (default: "text")

### Llama-Specific Parameters
- `--experiment_group`: Required for Llama to determine prompt format

## Troubleshooting

### Common Issues

1. **Checkpoint not found**: Make sure the checkpoint file exists and the path is correct
2. **Model path not found**: Update the model paths in `run_inference.py` with actual paths
3. **CUDA out of memory**: Reduce batch size or max_len
4. **Invalid checkpoint format**: Check that the checkpoint name follows the expected format

### Memory Optimization

For large datasets or limited GPU memory:
- Reduce `batch_size` (e.g., 8 or 16)
- Reduce `max_len` (e.g., 256 or 128)
- Use CPU by setting `CUDA_VISIBLE_DEVICES=""`

### Performance Tips

- Use larger batch sizes when possible for better GPU utilization
- For Llama models, batch size is typically 1 due to memory constraints
- Consider using mixed precision (fp16) for faster inference

## Notes

- The scripts use GPU 1 by default (`CUDA_VISIBLE_DEVICES="1"`)
- Llama models require the `experiment_group` parameter to generate appropriate prompts
- Checkpoint naming conventions must be followed for automatic parameter detection
- For Llama models, checkpoints are stored in directories rather than single files 