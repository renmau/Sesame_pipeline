import os
import glob
import yaml
import argparse
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as TorchFunctional
from pydantic import BaseModel
from typing import Tuple, List, Union
from pytorch_lightning.callbacks import StochasticWeightAveraging
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.dataset import Dataset
       
class Parameters(BaseModel):
    feature_columns:             List[str]
    label_columns:               List[str]
    hidden_layers:               List[int]
    trainingdata_csvfile_path:   "str"
    validationdata_csvfile_path: "str"
    testdata_csvfile_path:       "str"
    max_epochs:                  int
    batch_size:                  int
    nthreads:                    int
    activation_function:         str
    loss_function:               str
    output_dir:                  str
    learning_rate:               float
    weight_decay:                float

class Net(pl.LightningModule):
    def __init__(self, n_features, output_dim, learning_rate, weight_decay, hidden_layers, loss_function, activation_function, **kwargs):
        super(Net, self).__init__()
        self.learning_rate       = learning_rate
        self.weight_decay        = weight_decay
        self.n_hidden_layers     = len(hidden_layers)
        self.n_features          = n_features
        self.output_dim          = output_dim
        self.loss_function       = getattr(torch.nn, loss_function)()
        self.activation_function = getattr(TorchFunctional, activation_function)
        self.hidden_layers       = nn.ModuleList()
        self.activations         = nn.ModuleList()
        self.output_layer        = nn.Linear(in_features=hidden_layers[-1], out_features=self.output_dim)
        self.dropout             = nn.Dropout(p=0.0)
        for i in range(self.n_hidden_layers):
            layer = nn.Linear(in_features=self.n_features if i == 0 else hidden_layers[i - 1], out_features=hidden_layers[i])
            self.hidden_layers.append(layer)
        self.save_hyperparameters("n_features","output_dim","learning_rate","weight_decay","hidden_layers","loss_function","activation_function")
    def forward(self, x):
        for i, layer in enumerate(self.hidden_layers):
            x = self.activations[i](layer.forward(x)) if self.activations else self.activation_function(layer.forward(x))
        output = self.output_layer.forward(x)
        return output
    def custom_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x,)
        return self.loss_function(y_hat, y)
    def training_step(self, batch, batch_idx):
        loss = self.custom_step(batch=batch, batch_idx=batch_idx)
        return loss
    def test_step(self, batch, batch_idx):
        loss = self.custom_step(batch=batch, batch_idx=batch_idx)
        return loss
    def validation_step(self, batch, batch_idx):
        loss = self.custom_step(batch=batch, batch_idx=batch_idx)
        self.log("loss/val", loss)
        return loss
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=20, factor=0.1, min_lr=1.0e-6, verbose=True)
        return { "optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "loss/val", "interval": "epoch", "frequency": 1} }

class PyTorchDataset(torch.utils.data.Dataset):
    def __init__(self, features: Union[np.array, torch.tensor], labels: Union[np.array, torch.tensor]):
        self.features, self.labels = features, labels
    def __len__(self,) -> int:
        return len(self.features)
    def __getitem__(self, index: int) -> Tuple[torch.tensor, torch.tensor]:
        return (self.features[index, :], self.labels[index, :])

class PyTorchDataModule(pl.LightningDataModule):
    def __init__(self, params):
        super().__init__()
        self.feature_columns = params.feature_columns
        self.label_columns   = params.label_columns
        self.batch_size      = params.batch_size
        self.nthreads        = params.nthreads
        self.training_data   = self.load_csv_file(params.trainingdata_csvfile_path)
        self.test_data       = self.load_csv_file(params.testdata_csvfile_path)
        self.validation_data = self.load_csv_file(params.validationdata_csvfile_path)
    def load_csv_file(self, path) -> "Dataset":
        df = pd.read_csv(path)
        return PyTorchDataset(features=torch.from_numpy(df[self.feature_columns].to_numpy().astype(np.float32)), labels=torch.from_numpy(df[self.label_columns].to_numpy().astype(np.float32)))
    def train_dataloader(self,) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self.training_data, batch_size=self.batch_size, num_workers=self.nthreads, shuffle=True)
    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self.validation_data, num_workers=self.nthreads, batch_size=self.batch_size)
    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.nthreads)

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0., std=1.e-3)
        m.bias.data.fill_(0)

class EmulatorEvaluator:
    def __init__(self, model):
        self.model = model
    @classmethod
    def load(self, path):
        checkpointpath = [file for file in glob.glob(path + "/checkpoints/*.ckpt")][0]
        with open(path + "/hparams.yaml", "r") as fp:
            hyperparams = yaml.safe_load(fp)
        model = Net.load_from_checkpoint(checkpointpath, **hyperparams)
        return self(model=model)
    def __call__(self, inputs: Union[np.array, torch.tensor]): 
        inputs = torch.from_numpy(inputs.astype(np.float32))
        with torch.no_grad():
            predictions = self.model(inputs)
        return predictions.detach().numpy()

if __name__ == "__main__":
  with open("input.yaml", "r") as f:
      inputfile = yaml.safe_load(f)
  param      = Parameters(**inputfile)
  datamodule = PyTorchDataModule(param)
  model      = Net(n_features=datamodule.training_data.features.shape[-1], output_dim=datamodule.training_data.labels.shape[-1], **dict(param))
  model.apply(init_weights)
  callbacks  = [EarlyStopping(monitor="loss/val", patience=30, mode="min", verbose=False), StochasticWeightAveraging(1e-2)]
  trainer    = pl.Trainer(callbacks=callbacks, max_epochs=param.max_epochs, default_root_dir=param.output_dir, gradient_clip_val=0.5)
  os.makedirs(trainer.log_dir, exist_ok=True)
  with open(trainer.log_dir + "/input.yaml", "w") as f:
      yaml.dump(inputfile, f)
  trainer.fit(model=model, datamodule=datamodule)
  result     = trainer.test(model=model, datamodule=datamodule)
