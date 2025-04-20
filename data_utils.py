import torch
from torch.nn.utils.rnn import pad_sequence

USEABLE_KEYS = [i+":" for i in "KLMmNQRs"]
USELESS_KEYS = [i+":" for i in "ABCDFGHIOPrSTUVWXZ"]


def read_abc(path):
    keys = []
    notes = []
    with open(path) as rf:
        for line in rf:
            line = line.strip()
            if line.startswith("%"):
                continue

            if any([line.startswith(key) for key in USELESS_KEYS]):
                continue

            if any([line.startswith(key) for key in USEABLE_KEYS]):
                keys.append(line)
            else:
                notes.append(line)

    keys = " ".join(keys)
    notes = "".join(notes).strip()
    notes = notes.replace(" ", "")

    if notes.endswith("|"):
        notes = notes[:-1]

    notes = notes.replace("[", " [")
    notes = notes.replace("]", "] ")
    notes = notes.replace("(", " (")
    notes = notes.replace(")", ") ")
    notes = notes.replace("|", " | ")
    notes = notes.strip()
    notes = " ".join(notes.split(" "))
    
    if not keys or not notes:
        return None, None

    return keys, notes


def collate_function(batch):
    input_ids_list = [item[0] for item in batch] # item[0] is input_ids
    labels_list = [item[1] for item in batch]   # item[1] is labels
    
    # Pad the input sequences (encoder inputs)
    padded_input_ids = pad_sequence(
        input_ids_list, 
        batch_first=True, 
        padding_value=0
    )
    
    # Create attention mask for the encoder inputs
    attention_mask = (padded_input_ids != 0).long()
    
    # Pad the label sequences (decoder targets)
    # Pad with PAD_TOKEN_ID first
    padded_labels_temp = pad_sequence(
        labels_list, 
        batch_first=True, 
        padding_value=0 
    )
    # Clone and replace padding with IGNORE_INDEX so loss is not computed on padding
    padded_labels = padded_labels_temp.clone()
    padded_labels[padded_labels_temp == 0] = -100
    
    return {
        "input_ids": padded_input_ids,
        "attention_mask": attention_mask,
        "labels": padded_labels
    }
