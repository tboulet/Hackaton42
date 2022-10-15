import numpy as np
from tqdm.notebook import tqdm
import os
from os.path import join
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd

use_cuda = torch.cuda.is_available()
use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")

datasets_names = sorted(os.listdir(path='datasets'), key=lambda name: int(name[:2]))
datasets_names = {int(datasets_name.split("_")[0]): datasets_name for datasets_name in datasets_names}
print(datasets_names)










def load_datasets(n_dataset : int):
    dataset_name = join('datasets', datasets_names[n_dataset])
    X_labeled = np.load(join(dataset_name, "X_labeled.npy"))
    y_labeled = np.load(join(dataset_name, "y_labeled.npy"))
    X_unlabeled = np.load(join(dataset_name, "X_unlabeled.npy"))
    X_val = np.load(join(dataset_name, "X_val.npy"))
    
    
    # To tensor with device
    X_labeled = torch.from_numpy(X_labeled).to(device).float()
    y_labeled = torch.from_numpy(y_labeled).to(device).float()
    X_unlabeled = torch.from_numpy(X_unlabeled).to(device).float()
    X_val = torch.from_numpy(X_val).to(device).float()
    
    # Not usefull for now, we split after 
    # X_train, X_test, y_train, y_test = train_test_split(X_labeled, y_labeled)
    # X_train = torch.tensor(X_train, device=device).float()
    # X_test = torch.tensor(X_test, device=device).float()
    # y_train = torch.tensor(y_train, device=device)
    # y_test = torch.tensor(y_test, device=device)

    return X_labeled, y_labeled, X_unlabeled, X_val



def display_dataset(n_dataset : int, N_data : int = 10):
    if n_dataset == 0:
        X_labeled, y_labeled, X_unlabeled, X_val = load_datasets(0)
        print("Labeled ds shape: ", X_labeled.shape)
        print("Unlabeled ds shape: ", X_unlabeled.shape)
        print("Validation ds shape: ", X_val.shape)
        print("Labels shape: ", y_labeled.shape)
        print("Labeled ds labels: ", np.unique(y_labeled))
        print()
        # Display 5 points from the labeled dataset
        plt.plot(X_labeled[y_labeled == 0, 0][:N_data], X_labeled[y_labeled == 0, 1][:N_data], 'ob')
        plt.plot(X_labeled[y_labeled == 1, 0][:N_data], X_labeled[y_labeled == 1, 1][:N_data], 'or')
        plt.show()
        
        plt.plot(X_unlabeled[:N_data, 0], X_unlabeled[:N_data, 1], 'o')
        for i in range(2000):
            print(f"{i}: {X_val[i]}")
        plt.show()
        
        

def submit_results(model, uX_val, name : str):
    # Get (P(Xi=k))i,k
    with torch.no_grad():
        y_pred = model(uX_val)
    # Get (most_probable_class_of_i)i
    pred = torch.argmax(y_pred, dim=1)
    
    if len(pred.shape) == 1:    # for format such as toy (dataset 0), just round
        pred = y_pred.round().int()
    else:
        pass
    
    # Convert to np and then submit    
    pred = pred.cpu().numpy()
    df = pd.DataFrame(pred)
    df.to_csv(f"submission_{name}.csv", header=False, index=False)