import torch 
from torch import nn
import lightning as L
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small 
from torchmetrics.classification import BinaryAccuracy


def person_appereance_model(num_classes : int = 26): 
    model = mobilenet_v3_small() 
    model.classifier[3] = nn.Linear(in_features=1024, out_features=num_classes)
    return model  


class PersonAppearanceLightning(L.LightningModule): 
    def __init__(self, num_classes : int = 26, lr : float = 1e-5):
        super().__init__() 

        self.model = person_appereance_model(num_classes)
        self.lr = lr 

        self.train_accuracy = BinaryAccuracy()
        self.val_accuracy = BinaryAccuracy()

        self.save_hyperparameters()  

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)

        loss = F.binary_cross_entropy_with_logits(logits, y)
        acc = self.train_accuracy(logits, y)

        self.log("train_loss", loss, on_epoch=True, on_step=True)
        self.log("train_acc", acc, on_epoch=True, on_step=True)

        return loss 
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x) 

        loss = F.binary_cross_entropy_with_logits(logits, y)
        acc = self.val_accuracy(logits, y)

        self.log("val_loss", loss, on_epoch=True, on_step=True)
        self.log("val_acc", acc, on_epoch=True, on_step=True) 


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer