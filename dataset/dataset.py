import torch
import lightning as L
from PIL import Image
from torchvision.transforms import v2
from torch.utils.data import Dataset, DataLoader

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
    

class PersonAppereanceDataModule(L.LightningDataModule):
    def __init__(
            self,
            train_images : list[str], 
            train_labels : list[int], 
            valid_images : list[str], 
            valid_labels : list[int],
            batch_size : int = 32
        ) -> None: 
        super().__init__()

        self.train_images = train_images 
        self.train_labels = train_labels 
        self.valid_images = valid_images 
        self.valid_labels = valid_labels
        self.batch_size = batch_size

    def get_transform(self):
        transform = v2.Compose([
            v2.Resize((256, 192)),
            v2.ToTensor(),
        ]) 
        return transform  
    
    def setup(self, stage: str):
        transform = self.get_transform()
        self.train_dataset = PersonAppereanceDataset(self.train_images, self.train_labels, transform) 
        self.valid_dataset = PersonAppereanceDataset(self.valid_images, self.valid_labels, transform) 

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)
    
    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size)
