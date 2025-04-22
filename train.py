import torch
import pickle
import os
from datetime import datetime
from tqdm import tqdm
from model import get_model
from data_utils import read_abc, collate_function
from dataset import ABCDataset
import youtokentome as yttm
from transformers import Trainer, TrainingArguments, TrainerCallback, EncoderDecoderModel
from pathlib import Path

# --- Constants used in dataset preprocessing and callback ---
BOS_ID = 2 # Corresponds to config.decoder_start_token_id
EOS_ID = 3 # Corresponds to config.eos_token_id
PAD_ID = 0 # Assuming pad token id is 0
CONTEXT_BARS = 8 # Number of bars for context
TARGET_BARS = 8 # Number of bars for target
IGNORE_IDS = [BOS_ID, EOS_ID, PAD_ID] # IDs to ignore during decoding for display
MIN_TOKENS = 64 # Minimum number of tokens for a valid sequence
MAX_TOKENS = 500 # Maximum number of tokens for a valid sequence

class TestingCallback(TrainerCallback):
    def __init__(self, model, tokenizer, test_data, every_n_steps=100):
        self.model = model
        self.tokenizer = tokenizer
        # Ensure test_data is a list of (input_ids, labels) tuples
        self.test_data = test_data
        self.every_n_steps = every_n_steps
        self.output_dir = "test_outputs"
        os.makedirs(self.output_dir, exist_ok=True)

    def on_step_end(self, args, state, control, **kwargs):
        # Ensure model is on the correct device (especially if using multi-GPU)
        device = next(self.model.parameters()).device
        if state.global_step % self.every_n_steps == 0 and state.is_local_process_zero: # Only run on main process
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.output_dir, f"test_output_step_{state.global_step}_{timestamp}.txt")

            self.model.eval() # Set model to evaluation mode
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(f"Test Results at Step {state.global_step}\n")
                f.write("=" * 80 + "\n\n")

                with torch.no_grad(): # Disable gradient calculations
                    for i, (input_ids, labels) in enumerate(self.test_data):
                        # Move data to the correct device and add batch dimension
                        input_ids_batch = input_ids.unsqueeze(0).to(device)
                        labels_batch = labels # Keep labels as is for decoding target

                        # Get model predictions
                        pred_str, target_str, input_str = test_model(self.model, self.tokenizer, input_ids_batch, labels_batch)

                        # Write results
                        f.write(f"Example {i+1}\n")
                        f.write("-" * 40 + "\n")
                        f.write("Input Context:\n")
                        f.write(input_str + "\n\n")
                        f.write("Target:\n")
                        f.write(target_str + "\n\n")
                        f.write("Predicted:\n")
                        f.write(pred_str + "\n\n")
                        f.write("=" * 80 + "\n\n")
            self.model.train()


def load_train_data(filename="train_data.pkl"):
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except:
        return None


def test_model(model, tokenizer, input_ids_batch, labels):
    gen_tokens = model.generate(input_ids=input_ids_batch, 
                                max_length=MAX_TOKENS, 
                                min_length=MIN_TOKENS,
                                early_stopping=False,
                                # no_repeat_ngram_size=4,
                                # length_penalty=1.2,
                                repetition_penalty=1.1,
                                # Strategy 1: Beam search
                                # do_sample=False,
                                # num_beams=15,
                                # Strategy 2: Sampling
                                do_sample = True,
                                temperature = 0.7,
                                top_k = 50,
                                )

    # Decode generated tokens, input, and labels
    # Generated tokens might include decoder_start_token_id, remove special tokens
    pred_tokens = gen_tokens[0].tolist()
    pred_text = tokenizer.decode(pred_tokens, ignore_ids=IGNORE_IDS)[0] # Use global IGNORE_IDS

    # Labels are the target sequence, remove special tokens
    label_tokens = labels.tolist()
    target_text = tokenizer.decode(label_tokens, ignore_ids=IGNORE_IDS)[0] # Use global IGNORE_IDS

    # Input context is the original input_ids, remove special tokens
    input_context_tokens = input_ids_batch[0].tolist()
    input_context_text = tokenizer.decode(input_context_tokens, ignore_ids=IGNORE_IDS)[0]

    # Format the text (optional, adjust as needed)
    pred_text = pred_text.replace(" ", "").replace("|", "|\n")
    target_text = target_text.replace(" ", "").replace("|", "|\n")
    input_context_text = input_context_text.replace(" ", "").replace("|", "|\n")

    return pred_text, target_text, input_context_text


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    checkpoint = None

    training_args = TrainingArguments(
        output_dir="./ABCModel",
        overwrite_output_dir=True,
        num_train_epochs=20,
        per_device_train_batch_size=10,
        save_strategy="steps",
        save_steps=5000,
        save_total_limit=10,
        gradient_accumulation_steps=12,
        gradient_checkpointing=True,
        dataloader_num_workers=0,
        # label_smoothing_factor=0.1,
        learning_rate=1e-5,
        bf16=True,
        save_safetensors=False,
        dataloader_pin_memory=True,
        optim="adafactor",
        logging_dir="./logs",
        logging_strategy="steps",
        logging_steps=500,
    )

    tokenizer = yttm.BPE("abc.yttm")

    if checkpoint and os.path.isdir(checkpoint):
        print(f"Resuming training from checkpoint: {checkpoint}")
        model = EncoderDecoderModel.from_pretrained(checkpoint) 
    else:
        print("Initializing new model...")
        model = get_model(vocab_size=tokenizer.vocab_size())

    cached_data = load_train_data()

    if cached_data is not None:
        print("Using cached training data")
        train_data = cached_data
    else:
        print("Creating and preprocessing new training data...")
        train_paths = list(Path("./cleaned_data").glob("*.abc"))
        processed_data = []
        for p in tqdm(train_paths):
            keys, notes = read_abc(p)
            if keys is not None and notes is not None:
                keys_tokens = tokenizer.encode(keys, output_type=yttm.OutputType.ID) 

                notes_split = notes.split(" | ")
                notes_tokens_list = [tokenizer.encode(bar + " | ", output_type=yttm.OutputType.ID) for bar in notes_split] 
                flat_notes_tokens = [item for sublist in notes_tokens_list for item in sublist]
                sequence_len = len(flat_notes_tokens)

                if MIN_TOKENS < sequence_len < MAX_TOKENS and len(notes_tokens_list) >= (CONTEXT_BARS + TARGET_BARS): 
                    split_indx = CONTEXT_BARS 
                    context_notes_bars = notes_tokens_list[:split_indx]
                    target_notes_bars = notes_tokens_list[split_indx : split_indx + TARGET_BARS] 

                    context_notes_tokens = [item for sublist in context_notes_bars for item in sublist]
                    target_notes_tokens = [item for sublist in target_notes_bars for item in sublist]

                    
                    input_tokens = [BOS_ID] + keys_tokens + context_notes_tokens + [EOS_ID]
                    label_tokens = target_notes_tokens + [EOS_ID]

                    input_ids = torch.tensor(input_tokens, dtype=torch.long)
                    labels = torch.tensor(label_tokens, dtype=torch.long)
                    
                    processed_data.append((input_ids, labels))
               
        train_data = processed_data
        
        print("Saving preprocessed data...")
        with open("train_data.pkl", 'wb') as f:
            pickle.dump(train_data, f)

    print(f"Total training examples: {len(train_data)}")

    num_test_samples = 10
    test_data_for_callback = train_data[:num_test_samples]
    testing_callback = TestingCallback(model, tokenizer, test_data_for_callback, every_n_steps=5000) 

    train_dataset = ABCDataset(train_data)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_function,
        train_dataset=train_dataset,
        callbacks=[testing_callback]
    )

    trainer.train(resume_from_checkpoint=checkpoint) 


if __name__ == "__main__":
    main()