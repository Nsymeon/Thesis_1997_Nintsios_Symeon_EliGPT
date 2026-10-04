import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# 1. Ρυθμίσεις
model_id = "ilsp/Meltemi-7B-v1"
dataset_file = "dataset.jsonl" # Το αρχείο που φτιάξαμε
output_dir = "./meltemi-persona-weights"

# 2. Φόρτωση του Dataset
dataset = load_dataset('json', data_files=dataset_file, split='train')

# Συνάρτηση για να φτιάξουμε το Prompt format του Meltemi
def format_prompts(example):
    text = f"<|system|>\nΕίσαι μια περσόνα που βοηθάει προγραμματιστές να καταλάβουν την εμπειρία ενός ατόμου με όγκο. Απάντησε με ειλικρίνεια και συναίσθημα βασισμένος σε πραγματικές μαρτυρίες.</s>\n<|user|>\n{example['instruction']}</s>\n<|assistant|>\n{example['response']}</s>"
    return {"text": text}

dataset = dataset.map(format_prompts)

# 3. Ρύθμιση 4-bit Quantization (για να χωρέσει στη GPU)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# 4. Φόρτωση Μοντέλου και Tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# 5. Ρύθμιση LoRA (εκπαιδεύουμε μόνο ένα μικρό μέρος του μοντέλου)
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], # Στόχευση των layers του Meltemi
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 6. Παράμετροι Εκπαίδευσης
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3,          # Πόσες φορές θα δει τα δεδομένα (3-5 είναι καλά για αρχή)
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    optimizer="paged_adamw_32bit",
    save_steps=25,
    logging_steps=5,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="none"
)

# 7. Έναρξη Εκπαίδευσης
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_arguments,
)

print("Ξεκινάει η εκπαίδευση...")
trainer.train()

# 8. Αποθήκευση του εκπαιδευμένου "Adapter"
trainer.model.save_pretrained(output_dir)
print(f"Η εκπαίδευση ολοκληρώθηκε! Τα βάρη σώθηκαν στο {output_dir}")