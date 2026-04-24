import torch
import torch.nn as nn
import torch.optim as optim
import os
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, Subset, Dataset
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import numpy as np

#CNN architechture adapted from in class example as a starting architecture
class CNN(nn.Module):
    def __init__(self, device, num_classes=7, dropout_cnn=0.25, dropout_class=0.3, kernel_size=3, max_pool=2, padding=1):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(max_pool),
            nn.Dropout2d(p=dropout_cnn),
            nn.MaxPool2d(max_pool),
            nn.Conv2d(64, 128, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=dropout_cnn),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_class),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_class),
            nn.Linear(256, num_classes),

        )
        self.device = device

    def forward(self, x):
        cnn_output = self.cnn(x)
        output= self.classifier(cnn_output)
        return output
        
    def forward_feats(self, x):
        features = self.cnn(x)
        features = features.flatten(start_dim=1)
        return features

def train_one_epoch(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y, ids in train_loader:
        # print(type(y)
        # print(y.shape)
        # print(f'{idx+1}/{len(model.train_loader)}')
        x = x.to(model.device)
        y = y.to(model.device).long()
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
    
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        total_loss += loss.item() * x.size(0)
    avg_loss = total_loss / len(train_loader.dataset)
    acc = correct / total
    return avg_loss, acc

def run_experiment(model, train_loader, val_loader, num_epochs, lr=1e-3):
    # best_val_loss = 0
    criterion = nn.CrossEntropyLoss() #this loss converts the real values into probabilites in it first
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history = {"train_loss": [], "val_loss": [], "test_acc": []}
    best_f1_score = 0

    for epoch in range(1, num_epochs + 1):
        # print('Training epoch: ', epoch)
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        # print("Epoch " + str(epoch) + ' loss: ' + str(tr_loss))
        val_loss, macro_f1, val_info  = evaluate(model, val_loader, criterion)
        history["train_loss"].append(tr_loss)
        # history['train_accuracy'].append(tr_acc)
        history['val_loss'].append(val_loss)
        if macro_f1 > best_f1_score: 
            best_f1_score = macro_f1
            torch.save(model.state_dict(), "cnn_best.pt")
        # history["test_loss"].append(te_loss)
        # history["test_acc"].append(te_acc)
        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d} | train loss {tr_loss:.4f}  | train acc {tr_acc:.4f}")
            print(f"val loss {val_loss:.4f} | val_f1{macro_f1:.4f}")

            #| "
                #   f"test loss {te_loss:.4f} | test acc {te_acc:.3f}")
    return history, val_info

def evaluate(model, loader, criterion):
  model.eval()
  all_probs, all_preds, all_labels, all_ids = [], [], [], []
  val_loss = 0

  with torch.no_grad():
    for x, y, ids in loader:
        # print(ids)
        # print(type(y))
      x, y = x.to(model.device), y.to(model.device)
      logits = model(x)
      loss = criterion(logits, y)
      val_loss += loss.item() * x.size(0)

      probs = torch.softmax(logits, dim=1)
      preds = probs.argmax(dim=1)
    #   print(ids[:5])
    #   print(len(ids), probs.shape)

      all_probs.append(probs.cpu().numpy())
      all_preds.append(preds.cpu().numpy())
      all_labels.append(y.cpu().numpy())
      all_ids.extend(list(ids))

  val_loss = val_loss / len(loader.dataset)
  all_probs = np.concatenate(all_probs)
  all_preds = np.concatenate(all_preds)
  all_labels = np.concatenate(all_labels)
  all_ids = np.array(all_ids)
  macro_f1 = f1_score(all_labels, all_preds, average="macro")

  return val_loss, macro_f1, (all_probs, all_preds, all_labels, all_ids)


def predict_test(model, test_loader):
  model.eval()
  all_probs, all_preds, all_ids = [], [], []

  with torch.no_grad():
    for x, ids in test_loader:
      x = x.to(model.device)
      logits = model(x)
      probs = torch.softmax(logits, dim=1)
      preds = probs.argmax(dim=1)
      all_probs.append(probs.cpu().numpy())
      all_preds.append(preds.cpu().numpy())
      all_ids.extend(list(ids))

  all_probs = np.concatenate(all_probs)
  all_preds = np.concatenate(all_preds)
  all_ids = np.array(all_ids)

  return all_probs, all_preds, all_ids

# make dataset
class ImageDataset(Dataset):
  def __init__(self, X, ids, y = None):
    self.X = X
    self.y = y
    self.ids = ids

  def __len__(self):
    return len(self.X)

  def __getitem__(self, idx):
      
    if self.y is not None:
      return self.X[idx], self.y[idx], self.ids[idx]
    else: 
       return self.X[idx], self.ids[idx]

def build_dataloader(images, labels, ids):
    images = np.array(images)
    X_train, X_val, y_train, y_val, train_ids, val_ids = train_test_split(images, labels, ids, test_size=0.2, random_state=42, stratify=labels)
    X_train = torch.tensor(X_train, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    X_val = torch.tensor(X_val, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    y_train = torch.tensor(y_train)
    y_val = torch.tensor(y_val)

    train_dataset = ImageDataset(X_train, train_ids, y_train)
    val_dataset = ImageDataset(X_val, val_ids, y_val)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    return train_loader, val_loader

def build_testdata(images, ids):
   X_test = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0

   test_dataset = ImageDataset(X_test, ids)
   test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

   return test_loader


class FCN(nn.Module):
    def __init__ (self, criterion, train_loader, val_loader, device, lr=1e-3):
      super().__init__()
      self.fcn = nn.Sequential(
          nn.Flatten(),
          nn.Linear(3 * 224 * 224, 256),
          nn.BatchNorm1d(256),
          nn.ReLU(),
          nn.Linear(256, 128),
          nn.BatchNorm1d(128),
          nn.ReLU(),
          nn.Linear(128, 7),
      )
      self.criterion=criterion
      self.optimizer = optim.Adam(self.fcn.parameters(), lr=lr)
      self.train_loader = train_loader
      self.val_loader = val_loader
      self.device = device

    def forward(self, x):
      return self.fcn(x)

    def fit(self, epochs):
      history = {"train_loss": [], "test_loss": [], "test_acc": []}
      self.train()
      for epoch in range(1, epochs + 1):
        total_loss = 0
        for images, labels, _ in self.train_loader:
          # print(labels.size())
          images = images.to(self.device)
          labels = labels.to(self.device).long()
          # print(labels.size())
          probs = self(images)
          loss  = self.criterion(probs, labels)

          self.optimizer.zero_grad()
          loss.backward()
          self.optimizer.step()

          total_loss += loss.item() * labels.size(0)

        val_loss, probs, preds, labels, ids = evaluate(self, self.criterion)
        if epoch % 10 == 0:
          print("Epoch " + str(epoch) + ' loss: ' + str(total_loss / len(self.train_loader.dataset)) + ' val loss ' + str(val_loss))
        history["train_loss"].append(total_loss)

      avg_loss = total_loss / len(self.train_loader)
      print(f"Epoch {epoch:>2}/{epochs}  |  Loss = {avg_loss:.4f}")
      return history, avg_loss, val_loss, probs, preds, labels, ids
