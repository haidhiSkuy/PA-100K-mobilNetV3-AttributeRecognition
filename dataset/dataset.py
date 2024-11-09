import torch
from PIL import Image
from torch.utils.data import Dataset

class PersonAppereanceDataset(Dataset):
    def __init__(self, image_list, label_list, transform):
        self.image_list = image_list 
        self.label_list = label_list
        self.transform = transform

    def __len__(self):
        return len(self.image_list) 

    def __getitem__(self, index):
        image_path = self.image_list[index] 
        image = Image.open(image_path)
        
        label_arr = self.label_list[index] 
        label_tensor = torch.tensor(label_arr)

        if self.transform: 
            transformed = self.transform(image)

        return transformed, label_tensor.to(torch.float32)