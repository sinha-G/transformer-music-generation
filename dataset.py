from torch.utils.data import Dataset


class ABCDataset(Dataset):
    def __init__(self, data, is_test=False):
        self.data = data
        self.is_test = is_test
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
