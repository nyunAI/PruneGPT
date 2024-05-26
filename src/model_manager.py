import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from torch.distributed import init_process_group
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

class ModelManager:
    def __init__(self, model_name, fsdp_config=None):
        self.model_name = model_name
        self.fsdp_config = fsdp_config
        self.model = None
        self.tokenizer = None

    def load_model_and_tokenizer(self, quantize=False, bnb_config=None):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token ## For LLaMA3
        if quantize and bnb_config:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                torch_dtype=torch.float16,
                device_map="auto",
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.float16,device_map="auto")

    def export_model(self, save_directory):
        self.model.save_pretrained(save_directory)

    def get_tokenizer(self):
        return self.tokenizer

    def get_model(self):
        return self.model