import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import seaborn as sns


def visual_tabular():
    '''Visualizes tabular data with histograms and count plots.'''

    train_tab = np.load('data/train_tabular.npy')
    y = np.load('data/train_labels.npy')

    X = pd.DataFrame({'age':train_tab[:, 0], 'sex':train_tab[:, 1], 'anatomical_site':train_tab[:, 2]})
    X.hist()
    # plt.savefig(../figures/feature_distribution.png)
    plt.show()

    sns.countplot(x=y)
    plt.title("Class Balance")
    plt.xlabel("Target Label")
    plt.ylabel("Count")
    # plt.savefig(../figures/class_balance.png)
    plt.show()

# visual_tabular()

def process_data(tabular: bool, scale: bool, batch_size: int = 32, test_size: float = 0.2, random_state: int = 42):
    '''Preprocesses the data and returns DataLoaders for training and validation sets.
        with optional scaling for tabular features
    
    tabular (bool): whether to process tabular data or image data
    scale (bool): whether to scale the data or not
    batch_size (int): batch size for DataLoader
    test_size (float): proportion of data to use for train/val split
    random_state (int): set random seed for reproducibility
    
    Returns: 
        train_loader (DataLoader): DataLoader for training set
        val_loader (DataLoader): DataLoader for validation set'''

    y = np.load('data/train_labels.npy')

    if tabular: 
        X_data = np.load('data/train_tabular.npy')
        if scale:
             X_train, X_val, y_train, y_val = train_test_split(X_data, y, test_size=test_size, random_state=random_state, shuffle=True, stratify=y)
             scaler = StandardScaler()
             X_train = scaler.fit_transform(X_train)
             X_val = scaler.transform(X_val)
        
    
    else: 
        X =  np.load('data/train_images.npy').astype(np.float32)
        # norm pixel values from 0-255 to 0-1
        X /= np.max(X)
        # shape (train/test shape, 3, 224, 224)
        X_data = X.transpose(0, 3, 1, 2)
        
        X_train, X_val, y_train, y_val = train_test_split(X_data, y, test_size=test_size, random_state=random_state, shuffle=True, stratify=y)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)

    train_data = TensorDataset(X_train_tensor, X_val_tensor)
    val_data = TensorDataset(y_train_tensor, y_val_tensor)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader