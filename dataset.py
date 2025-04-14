import random
import torch
from torch.utils.data import Dataset


class ABCDataset(Dataset):
    def __init__(self, data, 
                 context_bars_num=8, 
                 target_bars_num=8,
                 bos_id=2,
                 eos_id=3,
                 is_test=False):
        
        self.notes = []
        self.keys = []

        for (keys, notes) in data:
            if notes is None:
                continue

            self.keys.append(keys)
            self.notes.append(notes)
        
        self.context_bars_num = context_bars_num
        self.target_bars_num = target_bars_num
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.is_test = is_test
        
    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, idx):
        notes = self.notes[idx]
        keys = self.keys[idx]
        
        if not self.is_test:
            split_indx = 8
            context_notes = notes[split_indx - self.context_bars_num : split_indx]
            target_notes = notes[split_indx: split_indx + self.target_bars_num]
        else:
            context_notes = notes
            target_notes = []

        # Flatten context_notes and target_notes
        context_notes = [item for sublist in context_notes for item in sublist]
        target_notes = [item for sublist in target_notes for item in sublist]

        # Prepare input tokens
        input_tokens = [self.bos_id] + keys + context_notes + [self.eos_id]

        # Prepare label tokens
        label_tokens = target_notes + [self.eos_id]
        
        input_ids = torch.tensor(input_tokens, dtype=torch.long)
        labels = torch.tensor(label_tokens, dtype=torch.long)

        # print("Raw input tokens:", input_tokens)
        # print("Raw label tokens:", label_tokens)

        return input_ids, labels