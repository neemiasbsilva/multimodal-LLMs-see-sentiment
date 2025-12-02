import torch
from transformers import (
    DistilBertModel,
    BertModel,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    AutoModelForMaskedLM,
    AutoModel
)
from trl import SFTTrainer
from peft import LoraModel, LoraConfig
import gc


class DistilBERTModel(torch.nn.Module):
    def __init__(self, bert_path):
        super().__init__()
        self.distilbert = DistilBertModel.from_pretrained(bert_path)
        # Output layer for Regression: Output size is 1 (a scalar score)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(768, 1)
        )

    def forward(self, ids, mask, token_type_id=None):
        output = self.distilbert(ids, attention_mask=mask)
        last_hidden_state = output.last_hidden_state
        CLS_token_state = last_hidden_state[:, 0, :]
        out = self.classifier(CLS_token_state)
        return out


class ModernBERTModel(torch.nn.Module):
    def __init__(self, model_id):
        super().__init__()
        self.model_id = model_id
        self.model = AutoModel.from_pretrained(model_id)

        # Output layer for Regression: Output size is 1
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.model.config.hidden_size, 1)
        )

    def forward(self, ids, mask, token_type_id=None):
        output = self.model(ids, attention_mask=mask)
        last_hidden_state = output.last_hidden_state

        # Use the CLS token (or the last hidden state) to pass through the classifier
        CLS_token_state = last_hidden_state[:, 0, :]
        out = self.classifier(CLS_token_state)

        return out


class Llama3:
    """
    Llama3 setup remains largely the same, but the prompting strategy 
    in the training script will change to request numerical scores.
    """
    def __init__(self, model_name):
        self.model_name = model_name
        self.compute_dtype = getattr(torch, "float16")
        self.model = None
        self.tokenizer = None

    def get_model(self):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=self.compute_dtype,
            bnb_4bit_use_double_quant=False,
            llm_int8_enable_fp32_cpu_offload=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            quantization_config=bnb_config,
            trust_remote_code=True,
        )

        self.model.config.use_cache = False
        self.model.config.pretraining_tp = 1

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        return self.model, self.tokenizer

    def cleanup(self):
        if self.model:
            del self.model
            self.model = None
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()