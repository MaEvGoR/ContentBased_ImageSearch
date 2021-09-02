import os
from pytorch_lightning.core import lightning
import torch
import time
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS
import pytorch_lightning as pl
from pipeline.clothes import TrainDatasetFolder
from pytorch_metric_learning.losses.soft_triple_loss import SoftTripleLoss
from pytorch_metric_learning.miners import TripletMarginMiner
from pytorch_metric_learning.distances import CosineSimilarity
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

class ClothesDataModule(pl.LightningDataModule):

    def __init__(self, data_dir:str = "Inshop-data/"):
        '''
        
        :param data_dir:
        '''
        super().__init__()
        self.data_dir = data_dir
        # self.transforms = transforms.Compose([CropClothes(detector.process), transforms.ToTensor(), transforms.Resize(size=(150, 150))])
        self.transforms = transforms.Compose([transforms.ToTensor(), transforms.Resize(size=(150, 150))])
    
    def prepare_data(self):
        # Put dataset download here
        train_data_dir = f'Inshop-data/train_data/'
        self.inshop_full = TrainDatasetFolder(train_data_dir, loader=default_loader, extensions=IMG_EXTENSIONS, transform=self.transforms)
        self.num_of_classes = len(self.inshop_full.classes)
    
    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:

            train_idx, val_idx = train_test_split(list(range(len(self.inshop_full))), test_size=0.1)

            self.inshop_train = Subset(self.inshop_full, train_idx)
            self.inshop_val = Subset(self.inshop_full, val_idx)
            

        # Assign test dataset for use in dataloader(s)
        # if stage == "test" or stage is None:
        #     self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transforms)

            # Optionally...
            # self.dims = tuple(self.mnist_test[0][0].shape)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.inshop_train, num_workers=8, batch_size=128)
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.inshop_val, batch_size=128)


datam = ClothesDataModule()
datam.prepare_data()

def ResNet50ImgEncoderConv(embeddings_size):
    resnet50 = torchvision.models.resnet50(pretrained=True)
    resnet50.fc = nn.Sequential(
        nn.Linear(2048, 1500),
        nn.Dropout(0.2),
        nn.Linear(1500, 800),
        nn.Dropout(0.2),
        nn.Linear(800, embeddings_size)
    )
    return resnet50
    # return nn.Sequential(
    #     nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=1),
    #     nn.MaxPool2d(kernel_size=2, stride=2),
    #     nn.Conv2d(6, 8, kernel_size=5, stride=1, padding=1),
    #     nn.MaxPool2d(kernel_size=2, stride=2),
    #     nn.Conv2d(8, 10, kernel_size=5, stride=1, padding=1),
    #     nn.MaxPool2d(kernel_size=2, stride=2))

# def SimpleClothesImgEncoderLinear():
#     return nn.Sequential(
#         nn.Linear(17 * 17 * 10, 1500), 
#         nn.Linear(1500, 800),
#         nn.Linear(800, embeddings_size))

def SimpleSoftmax(input_size, num_classes):
    return nn.Sequential(
        nn.Linear(input_size, num_classes),
        nn.Softmax(1)
    )

class ImgModel(pl.LightningModule):

    def __init__(self, loss, mining_funct, num_classes, embeddings_size):
        super().__init__()

        self.conv = ResNet50ImgEncoderConv(embeddings_size)
        # self.linear = SimpleClothesImgEncoderLinear(embeddings_size)
        self.decoder = SimpleSoftmax(embeddings_size, num_classes)
        self.loss = loss
        self.mining_funct = mining_funct
    
    def forward(self, sample):

        x = self.conv(sample)
        x = x.view(x.size(0), -1)
        embeddings = self.linear(x)

        return embeddings
    
    def training_step(self, batch, batch_idx):
        
        x, y = batch

        conv_features = self.conv(x)
        embeddings = conv_features.view(conv_features.size(0), -1)

        classes = self.decoder(embeddings)

        classes = torch.argmax(classes, dim=1)

        indices = self.mining_funct(embeddings, y)

        loss = self.loss.compute_loss(embeddings, classes, indices)
        # self.loss_optimizer.step()
        
        loss = torch.sum(loss['loss']['losses'])

        self.log("train_loss", loss)
        
        return loss
    
    def on_epoch_end(self) -> None:
        self.loss_optimizer.step()
        return super().on_epoch_end()
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.1)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1)

        self.loss_optimizer = torch.optim.SGD(self.loss.parameters(), lr=0.01)

        return [optimizer], [lr_scheduler]

loss = SoftTripleLoss(num_classes=datam.num_of_classes, embedding_size=300).to(device)
mining_func = TripletMarginMiner(margin = 0.2, distance = CosineSimilarity(), type_of_triplets = "semihard")

model = ImgModel(loss=loss, mining_funct=mining_func, num_classes=datam.num_of_classes, embeddings_size=300)
trainer = pl.Trainer(max_epochs=10)
trainer.fit(model, datamodule=datam)

