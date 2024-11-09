import os
import wandb
import lightning as L
from scipy.io import loadmat
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from dataset.dataset import PersonAppereanceDataModule
from model.person_appereance_model import PersonAppearanceLightning 

checkpoint = ModelCheckpoint(
    monitor='val_loss',
    dirpath='checkpoint',
    filename='apperence-{epoch:02d}-{val_loss:.2f}', 
    save_top_k=1, 
    save_on_train_epoch_end=True,
    verbose=True, 
    save_weights_only=True
)


wandb.login(key="06ee7ca7307838ddb249c4cda6662d79e7d7d16d")  
wandb_logger = WandbLogger(project="testing", log_model=True) 

annotation = "/kaggle/working/dataset/PA-100K/annotation.mat"
mat_data = loadmat(annotation)

root_dir = "/kaggle/working/dataset/PA-100K/images" 

labels = mat_data['attributes']
labels = [str(i[0][0]) for i in labels]

# Training Data
train_images = mat_data['train_images_name']
train_images = [str(i[0][0]) for i in train_images]
train_images = [os.path.join(root_dir, i) for i in train_images] 

train_labels = mat_data['train_label']
train_labels = [list(i) for i in train_labels]

# Validation Data
valid_images = mat_data['test_images_name'] 
valid_images = [str(i[0][0]) for i in valid_images]
valid_images = [os.path.join(root_dir, i) for i in valid_images] 

valid_labels = mat_data['test_label']
valid_labels = [list(i) for i in valid_labels]  

# Training
model = PersonAppearanceLightning(8)
data = PersonAppereanceDataModule(train_images, train_labels, valid_images, valid_labels, 16)
trainer = L.Trainer(max_epochs=20, logger=wandb_logger, callbacks=[checkpoint], accelerator="gpu", devices="auto")

trainer.fit(model, data)