# -*- coding: utf-8 -*-
import os
import shutil
import pprint
import pandas as pd
import os
import torch
import numpy as np
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import torch.optim
import torchmetrics
#cross-validation
from sklearn.model_selection import KFold
from torch.utils.data import Subset
from sklearn.model_selection import TimeSeriesSplit
import random
from torchinfo import summary
import argparse





def transform_data(X, y):
  #preprocessing
  scaler_X= MinMaxScaler()
    #train_features
  X_scaled=scaler_X.fit_transform(X)

  #convert to dataframe
  X_scaled=pd.DataFrame(X_scaled, columns=X.columns)

  y_scaled=y.values.reshape(-1,1)

  return X_scaled, y_scaled

def create_sequence_df(X, y, seq_length=24):
  xs, ys = [], []
  for i in range(len(X)-seq_length):
    xs.append(X[i:i+seq_length])
    ys.append(y[i+seq_length])

  return np.array(xs), np.array(ys)

class EnergyDataset(Dataset):

  def __init__(self, csv_path, transform, create_sequence):

    super().__init__()
    self.data=pd.read_csv(csv_path, index_col='DateCET/CEST')
    self.features = ['price_energy[EUR/MWh]', 'is_peak', 'load_energy[MW]', 'wind_energy_generation[MWh]', 'is_holiday', 'price_carbon_permits[EUR]',
       'price_gas[EUR/m3]', 'pv_energy_generation[MWh]', 'price_coal[EUR]']
    self.target = 'price_energy[EUR/MWh]'
    self.seq_length=24
    self.transform=transform
    self.create_sequence=create_sequence

    X = self.data.loc[:, self.features].astype('float32')
    y = self.data.loc[:, self.target].astype('float32')

    X_scaled, y_scaled = transform(X, y)

    X_seq, y_seq = create_sequence(X_scaled, y_scaled, seq_length=self.seq_length)

    #tensorDataset
    self.dataset=TensorDataset(
        torch.tensor(X_seq, dtype=torch.float32),
        torch.tensor(y_seq, dtype=torch.float32))

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    return self.dataset[idx]

#LSTM
class Net(nn.Module):
  def __init__(self, input_size, hidden_size, device):
    super().__init__()
    self.hidden_size=hidden_size
    self.device=device
    self.lstm = nn.LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=2,
        batch_first=True,
    )

    self.fc=nn.Linear(hidden_size,1)
    self.dropout=nn.Dropout(0.2)

  def forward(self, x):
    h0 = torch.zeros(2, x.size(0), self.hidden_size).to(self.device)
    c0 = torch.zeros(2, x.size(0), self.hidden_size).to(self.device)
    out, _ = self.lstm(x, (h0, c0))
    out = self.fc(out[:, -1, :])
    out = self.dropout(out)
    return out

class EarlyStopping:
    def __init__(self, patience, delta, verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.best_loss = None
        self.no_improvement_count = 0
        self.stop_training = False

    def check_early_stop(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.no_improvement_count = 0
            print(f'Validation improved')

        else:
            self.no_improvement_count += 1
            print(f"No improvement ({self.no_improvement_count}/{self.patience})")

            if self.no_improvement_count >= self.patience:
                self.stop_training = True
                if self.verbose:
                    print("Stopping early as no improvement has been observed.")

#class with training+validation+test
class Trainer:
  def __init__(self, model, criterion, optimizer, device, input_size, fold):
    self.model = model
    self.criterion = criterion
    self.optimizer = optimizer
    self.device = device
    self.input_size = input_size
    self.fold = fold

  def train(self, dataloader_train, epoch):
    self.model.train()

    running_loss = 0.0
    for batch_idx, batch in enumerate(dataloader_train):
      inputs, labels = batch[0].to(self.device), batch[1].to(self.device)
      self.optimizer.zero_grad()
      #prediction
      outputs = self.model(inputs)

      loss = self.criterion(outputs, labels)
      #backpropagation
      loss.backward()
      self.optimizer.step()

      running_loss += loss.item()

    avg_loss = running_loss/len(dataloader_train)
    return running_loss/len(dataloader_train)


  def validate(self, dataloader_val, epoch):
    self.model.eval()

    self.model.eval()
    val_loss = 0.0
    with torch.no_grad():
      for batch in dataloader_val:
        inputs, labels = batch[0].to(self.device), batch[1].to(self.device)
        outputs = self.model(inputs)
        outputs=outputs.reshape(-1,1)
        val_loss += self.criterion(outputs, labels).item()

    avg_val_loss = val_loss/len(dataloader_val)
    return val_loss/len(dataloader_val)

  def plot_losses(self, train_losses, val_losses):
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='val')
    plt.title('model_loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig(f"fold_{self.fold+1}_loss_curve.png")

  def plot_residuals(y_true, y_pred):
    residuals = y_pred - y_true

      # Scatter plot: residuals vs true values
    plt.figure(figsize=(8,5))
    plt.scatter(y_true, residuals, alpha=0.6)
    plt.axhline(0, color='red', linestyle='--', linewidth=1)
    plt.xlabel("True values")
    plt.ylabel("Residuals (Predicted - True)")
    plt.title("Residuals vs True values")
    plt.show()

    # Histogram of residuals
    plt.figure(figsize=(8,5))
    plt.hist(residuals, bins=30, edgecolor='k', alpha=0.7)
    plt.axvline(0, color='red', linestyle='--', linewidth=1)
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.title("Distribution of Residuals")
    plt.show()

  def fit(self, dataloader_train, dataloader_val, epochs, early_stopping, batch):

    train_losses = []
    val_losses = []
    for epoch in range(epochs):
      #train
      avg_running_loss=self.train(dataloader_train, epoch)
      #validate
      avg_val_loss=self.validate(dataloader_val, epoch)
      train_losses.append(avg_running_loss)
      val_losses.append(avg_val_loss)

      #check early stopping condition
      early_stopping.check_early_stop(avg_val_loss)
      if early_stopping.stop_training:
        print(f"Early stopping  at epoch{epoch}.")
        break


      print(f"Epoch {epoch +1}/{epochs}, Train Loss: {avg_running_loss}, Validation Loss: {avg_val_loss}")

    self.plot_losses(train_losses, val_losses)

    return self.model

#----TimeCrossValidator ----
class TimeCrossValidator:
  def __init__(self, dataset_train, dataset_test, k_folds, hidden_size, learning_rate, device, epochs, batch_size, patience=3, delta=0.01):
    self.dataset_train = dataset_train
    self.dataset_test = dataset_test
    self.k_folds = k_folds
    self.hidden_size = hidden_size
    self.learning_rate = learning_rate
    self.device = device
    self.epochs = epochs
    self.batch_size = batch_size
    self.patience = patience
    self.delta = delta
    self.results = {}
    self.model_paths = [] # Added model_paths attribute

  def run_cross_validation(self):
    tscv = TimeSeriesSplit(n_splits=self.k_folds)
    input_size = self.dataset_train[0][0].shape[1]

    fold_models = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(self.dataset_train)):
      print(f"Fold {fold +1}/{self.k_folds}")

      train_subset = Subset(self.dataset_train, train_idx)
      val_subset = Subset(self.dataset_train, val_idx)

      dataloader_train = DataLoader(train_subset, batch_size=self.batch_size, shuffle=False)
      dataloader_val = DataLoader(val_subset, batch_size=self.batch_size, shuffle=False)

      model = Net(input_size=input_size, hidden_size=self.hidden_size, device=self.device).to(self.device)
      criterion = nn.MSELoss()
      optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
      early_stopping = EarlyStopping(patience=self.patience, delta=self.delta, verbose=True)
      trainer = Trainer(model, criterion, optimizer, self.device, input_size, fold)

      trained_model = trainer.fit(dataloader_train, dataloader_val, self.epochs, early_stopping, self.batch_size)


      self.results[fold+1] = early_stopping.best_loss
      fold_models.append(trained_model)

      #save model locally
      model_path = f"{fold+1}_model.pth"
      torch.save(trained_model.state_dict(), model_path)
      self.model_paths.append(model_path)
      print("")

    print("Cross-validation done. Building ensemble model...")
    return  self.model_paths

class EnsembleModel(nn.Module):
  def __init__(self, model_paths, input_size, hidden_size, device):
    super().__init__()
    self.model_paths = model_paths
    self.device = device
    self.models = []

    # Load fold models into memory
    for path in model_paths:
      model = Net(input_size=input_size, hidden_size=hidden_size, device=device).to(device)
      model.load_state_dict(torch.load(path, map_location=device))
      model.eval()
      self.models.append(model)

  def forward(self, x):
    preds_sum = torch.zeros((x.size(0), 1), device=self.device)  # adjust output size
    for model in self.models:
      preds_sum += model(x)
    return preds_sum / len(self.models)

class EvaluatorEnsemble():
  def __init__(self, dataset_train, df_test, batch_size, epochs, device, model_paths, hidden_size):
    self.df_test = df_test
    self.batch_size = batch_size
    self.epochs = epochs
    self.device = device
    self.model_paths = model_paths
    self.hidden_size = hidden_size
    self.dataset_train =  dataset_train



    def build_ensemble(self):
      '''
      The function build ensemble model from models after time cross validation.
      '''
    input_size = self.dataset_train[0][0].shape[1]

    ensemble_model = EnsembleModel(self.model_paths, input_size,
                                       self.hidden_size, self.device)
    self.test_ensemble(ensemble_model)
    
    return ensemble_model

  def calculate_metrics(self, preds, labels):
    '''
    The function calculates metrics.
    '''
    mse = torchmetrics.MeanSquaredError()
    mae = torchmetrics.MeanAbsoluteError()
    rmse = torchmetrics.MeanSquaredError(squared=False)

    mse.update(preds, labels)
    mae.update(preds, labels)
    rmse.update(preds, labels)

    print(f"MSE: {mse.compute()}")
    print(f"MAE: {mae.compute()}")
    print(f"RMSE: {rmse.compute()}")



  def test_ensemble(self, ensemble_model):

    all_preds = []
    all_labels = []

    dataloader_test = DataLoader(self.df_test, self.batch_size, shuffle=False, num_workers=2)


    ensemble_model.eval()
    with torch.no_grad():
      for X, y in dataloader_test:
        X, y = X.to(self.device), y.to(self.device)
        preds = ensemble_model(X)
        preds=preds.reshape(-1,1)

        all_preds.append(preds)
        all_labels.append(y)

    self.calculate_metrics(preds, y)

def main(args):
  dataset_train = EnergyDataset(args.train_path, transform_data, create_sequence_df)
  dataset_test = EnergyDataset(args.test_path, transform_data, create_sequence_df)
  print("Creating datasets is completed.")

  cv_args = {
        "k_folds": args.folds,
        "hidden_size": args.hidden_size,
        "learning_rate": args.learning_rate,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
}


  tscv = TimeCrossValidator(
      dataset_train=dataset_train,
      dataset_test=dataset_test,
      **cv_args
)

  print("Starting time series crossvalidation...")
  model_paths = tscv.run_cross_validation()

  ensemble_evaluator = EvaluatorEnsemble(
      dataset_train=dataset_train,
      model_paths=model_paths,
      df_test=dataset_test,
      batch_size=args.batch_size,
      epochs=args.epochs,
      device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
      hidden_size=args.hidden_size
  )

  
  ensemble_model = ensemble_evaluator.build_ensemble()
  torch.save({"model_state": ensemble_model.state_dict(),
		"hidden_size" : args.hidden_size,
		"epochs" : args.epochs,
		"batch_size" : args.batch_size},
 		args.output_path)

  print("Testing model...")
  ensemble_evaluator.test_ensemble(ensemble_model)

  

if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='Energy Price Forecasting')

  parser.add_argument("-train_path", "--train_path", type=str, help="Path to the train dataset")
  parser.add_argument("-test_path", "--test_path", type=str, help="Path to the test dataset")
  parser.add_argument("-f", "--folds", type=int, default=3, help="Number of folds for cross-validation")
  parser.add_argument("-hs", "--hidden_size", type=int, default=200, help="Number of hidden units in the LSTM layer")
  parser.add_argument("-e", "--epochs", type=int, default=50, help="Number of training epochs")
  parser.add_argument("-lr", "--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer")
  parser.add_argument("-bs", "--batch_size", type=int, default=64, help="Batch size for training")
  parser.add_argument("-out_path", "--output_path", type=str, help="Path to save the trained model")

  args = parser.parse_args()

  main(args)