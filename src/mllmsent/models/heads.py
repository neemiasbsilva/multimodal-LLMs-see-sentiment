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
    def __init__(self, bert_path, class_size):
        super().__init__()
        self.distilbert = DistilBertModel.from_pretrained(bert_path)
        # print(self.distilbert)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(768, class_size)
            # torch.nn.Linear(768, 1024),
            # torch.nn.ReLU(),
            # torch.nn.Linear(1024, class_size),
        )

    def forward(self, ids, mask, token_type_id):
        output = self.distilbert(ids, attention_mask=mask)
        last_hidden_state = output.last_hidden_state
        CLS_token_state = last_hidden_state[:, 0, :]
        out = self.classifier(CLS_token_state)
        return out


class BERTFinetuningModel(torch.nn.Module):
    def __init__(self, bert_path):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(1024, 1024),
            torch.nn.BatchNorm1d(1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(1024, 768),
            torch.nn.ReLU(),
            torch.nn.Linear(768, 3),
        )

    def forward(self, ids, mask, token_type_id):
        output = self.bert(ids, mask, token_type_id)
        last_hidden_state = output.last_hidden_state
        CLS_token_state = last_hidden_state[:, 0, :]
        out = self.classifier(CLS_token_state)
        return out


class RoBERTaFinetuningModel(torch.nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.roberta = AutoModelForSequenceClassification.from_pretrained(model_path)

    def forward(self, ids, mask, token_type_id):
        return self.roberta(ids, mask).logits


class Llama3:
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
            # cache_dir="/mnt/raid5/neemias/cache-dir"
        )

        self.model.config.use_cache = False
        self.model.config.pretraining_tp = 1

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            # cache_dir="/mnt/raid5/neemias/cache-dir"
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        return self.model, self.tokenizer

    def cleanup(self):
        # Delete model and tokenizer references
        if self.model:
            del self.model
            self.model = None
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None

        # Clear CUDA cache if on GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Run garbage collection
        gc.collect()


class ModernBERTModel(torch.nn.Module):
    def __init__(self, model_id, class_size):
        super().__init__()
        self.model_id = model_id
        self.model = AutoModel.from_pretrained(model_id)

        # Add a custom classifier on top of the pre-trained model for fine-tuning
        self.classifier = torch.nn.Sequential(
            # torch.nn.Linear(self.model.config.hidden_size, class_size)
            torch.nn.Linear(self.model.config.hidden_size, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, class_size),
        )

    def forward(self, ids, mask, token_type_id=None):
        # Pass the input through the pre-trained model
        output = self.model(ids, attention_mask=mask)
        last_hidden_state = output.last_hidden_state

        # Use the CLS token (or the last hidden state) to pass through the classifier
        CLS_token_state = last_hidden_state[:, 0, :]
        out = self.classifier(CLS_token_state)

        return out
