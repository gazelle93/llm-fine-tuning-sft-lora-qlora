from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from config import Config

def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer

def load_model_with_sft():
    model = AutoModelForCausalLM.from_pretrained(
        Config.MODEL_NAME,
        device_map="auto"
    )

    model.print_trainable_parameters()

    return model

def load_model_with_lora(load_in_4bit=True, load_in_8bit=False):
    model = AutoModelForCausalLM.from_pretrained(
        Config.MODEL_NAME,
        device_map="auto"
    )

    lora_config = LoraConfig(
        r=Config.LORA_R,
        lora_alpha=Config.LORA_ALPHA,
        target_modules=Config.TARGET_MODULES,
        lora_dropout=Config.LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model

def load_model_with_qlora(load_in_4bit=True, load_in_8bit=False, double_quant=True, compute_dtype="float16", quant_type="nf4"):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        bnb_4bit_use_double_quant=double_quant,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=quant_type
    )

    model = AutoModelForCausalLM.from_pretrained(
        Config.MODEL_NAME,
        device_map="auto",
        quantization_config=bnb_config
    )

    lora_config = LoraConfig(
        r=Config.LORA_R,
        lora_alpha=Config.LORA_ALPHA,
        target_modules=Config.TARGET_MODULES,
        lora_dropout=Config.LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model
