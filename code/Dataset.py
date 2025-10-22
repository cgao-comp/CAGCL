import torch
import numpy as np
import pandas as pd
import json
import os
from transformers import pipeline
from datetime import datetime as dt
from torch.utils.data import Dataset
from tqdm import tqdm
#from keras.preprocessing.sequence import pad_sequences
import networkx as nx


class Twibot20:
    def __init__(self, root='./Data/Twibot20/', device='cpu', process=True, save=True):
        self.root = root
        self.device = device
        self.process = process
        self.save = save

        if not self.root.endswith('/'):
            self.root += '/'

        print(f"Loading Twibot20 dataset from {self.root}")

    def dataloader1(self):
        print('Loading Twibot20 dataset...')

        label_path = self.root + 'label.pt'
        print('Loading labels...', end='   ')
        if os.path.exists(label_path):
            labels = torch.load(label_path).to(self.device)
            print('Done')
        else:
            raise FileNotFoundError(f"Label file not found: {label_path}")

        des_path = self.root + 'des_tensor.pt'
        tweets_path = self.root + 'tweets_tensor.pt'
        print('Loading textual features...', end='   ')
        if os.path.exists(des_path) and os.path.exists(tweets_path):
            des_tensor = torch.load(des_path).to(self.device)
            tweets_tensor = torch.load(tweets_path).to(self.device)
            print('Done')
        else:
            raise FileNotFoundError(f"Missing text feature files: {des_path if not os.path.exists(des_path) else ''} {tweets_path if not os.path.exists(tweets_path) else ''}")

        num_prop_path = self.root + 'num_properties_tensor.pt'
        print('Loading numerical features...', end='   ')
        if os.path.exists(num_prop_path):
            num_prop = torch.load(num_prop_path).to(self.device)
            print('Done')
        else:
            raise FileNotFoundError(f"Numerical features file not found: {num_prop_path}")

        cat_prop_path = self.root + 'cat_properties_tensor.pt'
        print('Loading categorical features...', end='   ')
        if os.path.exists(cat_prop_path):
            cat_prop = torch.load(cat_prop_path).to(self.device)
            print('Done')
        else:
            raise FileNotFoundError(f"Categorical features file not found: {cat_prop_path}")

        edge_index_path = self.root + 'edge_index.pt'
        edge_type_path = self.root + 'edge_type.pt'
        print('Loading graph structure...', end='   ')
        if os.path.exists(edge_index_path) and os.path.exists(edge_type_path):
            edge_index = torch.load(edge_index_path).to(self.device)
            edge_type = torch.load(edge_type_path).to(self.device)

            mask = (edge_type == 0) | (edge_type == 1)
            edge_index = edge_index[:, mask]
            edge_type = edge_type[mask]

            print(f'Done (edges: {edge_index.size(1)}, types: {edge_type.unique().size(0)})')
        else:
            raise FileNotFoundError(f"Missing graph files: {edge_index_path if not os.path.exists(edge_index_path) else ''} {edge_type_path if not os.path.exists(edge_type_path) else ''}")

        print('Splitting dataset...')
        train_idx = torch.arange(0, 8278).to(self.device)
        val_idx = torch.arange(8278, 8278 + 2365).to(self.device)
        test_idx = torch.arange(8278 + 2365, 8278 + 2365 + 1183).to(self.device)
        print(f'Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}')

        print('Twibot20 dataset loaded successfully!')
        return des_tensor, tweets_tensor, num_prop, cat_prop, edge_index, edge_type, labels, train_idx, val_idx, test_idx



class Cresci15:
    def __init__(self, root='./Data/Cresci15/', device='cpu', process=True, save=True):
        self.root = root
        self.device = device
        self.process = process
        self.save = save

        if not self.root.endswith('/'):
            self.root += '/'

        print(f"Loading Cresci15 dataset from {self.root}")

    def dataloader1(self):
        print('Loading Cresci15 dataset...')

        label_path = self.root + 'label.pt'
        print('Loading labels...', end='   ')
        if os.path.exists(label_path):
            labels = torch.load(label_path).to(self.device)
            print('Done')
        else:
            raise FileNotFoundError(f"Label file not found: {label_path}")

        des_path = self.root + 'des_tensor.pt'
        tweets_path = self.root + 'tweets_tensor.pt'
        print('Loading textual features...', end='   ')
        if os.path.exists(des_path) and os.path.exists(tweets_path):
            des_tensor = torch.load(des_path).to(self.device)
            tweets_tensor = torch.load(tweets_path).to(self.device)
            print('Done')
        else:
            missing = []
            if not os.path.exists(des_path):
                missing.append(des_path)
            if not os.path.exists(tweets_path):
                missing.append(tweets_path)
            raise FileNotFoundError(f"Missing text feature files: {', '.join(missing)}")

        num_prop_path = self.root + 'num_properties_tensor.pt'
        print('Loading numerical features...', end='   ')
        if os.path.exists(num_prop_path):
            num_prop = torch.load(num_prop_path).to(self.device)
            print('Done')
        else:
            raise FileNotFoundError(f"Numerical features file not found: {num_prop_path}")

        cat_prop_path = self.root + 'cat_properties_tensor.pt'
        print('Loading categorical features...', end='   ')
        if os.path.exists(cat_prop_path):
            category_prop = torch.load(cat_prop_path).to(self.device)
            print('Done')
        else:
            raise FileNotFoundError(f"Categorical features file not found: {cat_prop_path}")

        edge_index_path = self.root + 'edge_index.pt'
        edge_type_path = self.root + 'edge_type.pt'
        print('Loading graph structure...', end='   ')
        if os.path.exists(edge_index_path) and os.path.exists(edge_type_path):
            edge_index = torch.load(edge_index_path).to(self.device)
            edge_type = torch.load(edge_type_path).to(self.device)

            mask = (edge_type == 0) | (edge_type == 1)
            edge_index = edge_index[:, mask]
            edge_type = edge_type[mask]

            print(f'Done (edges: {edge_index.size(1)}, types: {edge_type.unique().size(0)})')
        else:
            missing = []
            if not os.path.exists(edge_index_path):
                missing.append(edge_index_path)
            if not os.path.exists(edge_type_path):
                missing.append(edge_type_path)
            raise FileNotFoundError(f"Missing graph files: {', '.join(missing)}")

        train_idx_path = self.root + 'train_idx.pt'
        val_idx_path = self.root + 'val_idx.pt'
        test_idx_path = self.root + 'test_idx.pt'
        print('Loading data splits...', end='   ')

        if os.path.exists(train_idx_path) and os.path.exists(val_idx_path) and os.path.exists(test_idx_path):
            train_idx = torch.load(train_idx_path).to(self.device)
            val_idx = torch.load(val_idx_path).to(self.device)
            test_idx = torch.load(test_idx_path).to(self.device)
            print('Done')
        else:
            missing = []
            if not os.path.exists(train_idx_path):
                missing.append(train_idx_path)
            if not os.path.exists(val_idx_path):
                missing.append(val_idx_path)
            if not os.path.exists(test_idx_path):
                missing.append(test_idx_path)
            raise FileNotFoundError(f"Missing split files: {', '.join(missing)}")

        print('Cresci15 dataset loaded successfully!')
        print(f'Total nodes: {len(labels)}')
        print(f'Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}')

        return des_tensor, tweets_tensor, num_prop, category_prop, edge_index, edge_type, labels, train_idx, val_idx, test_idx
    


class MGTAB:
    def __init__(self, root='./Data/MGTAB/', device='cpu', process=True, save=True):
        self.root = root
        self.device = device
        self.process = process
        self.save = save

        if not self.root.endswith('/'):
            self.root += '/'

        print(f"Loading MGTAB dataset from {self.root}")

    def dataloader1(self):
        print('Loading MGTAB dataset...')

        label_path = self.root + 'labels_bot.pt'
        print('Loading labels...', end='   ')
        if os.path.exists(label_path):
            labels = torch.load(label_path).to(self.device)
            print('Done')
        else:
            raise FileNotFoundError(f"Label file not found: {label_path}")

        features_path = self.root + 'features.pt'
        combined_features = torch.load(features_path).to(self.device)
        feature_dim = combined_features.size(1)

        print('Loading tweet features...', end='   ')
        tweets_tensor = combined_features[:, feature_dim-768:].to(self.device)

        print('Loading numerical features...', end='   ')
        num_indices = torch.tensor([4, 6, 7, 8, 10, 11, 12, 13, 14, 15], device=self.device)
        num_prop = combined_features[:, num_indices].to(self.device)

        print('Loading categorical features...', end='   ')
        cat_indices = torch.tensor([1, 2, 3, 5, 9, 16, 17, 18, 19, 20], device=self.device)
        category_prop = combined_features[:, cat_indices].to(self.device)

        edge_index_path = self.root + 'edge_index.pt'
        edge_type_path = self.root + 'edge_type.pt'
        print('Loading graph structure...', end='   ')
        if os.path.exists(edge_index_path) and os.path.exists(edge_type_path):
            edge_index = torch.load(edge_index_path).to(self.device)
            edge_type = torch.load(edge_type_path).to(self.device)

            mask = (edge_type == 0) | (edge_type == 1)
            edge_index = edge_index[:, mask]
            edge_type = edge_type[mask]

            print(f'Done (edges: {edge_index.size(1)}, types: {edge_type.unique().size(0)})')
        else:
            raise FileNotFoundError(f"Missing graph files: {edge_index_path if not os.path.exists(edge_index_path) else ''} {edge_type_path if not os.path.exists(edge_type_path) else ''}")

        print('Splitting dataset (7:2:1)...', end='   ')
        dataset_size = len(labels)
        train_size = int(dataset_size * 0.7)
        val_size = int(dataset_size * 0.2)
        test_size = dataset_size - train_size - val_size

        indices = torch.arange(dataset_size, device=self.device)
        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size + val_size]
        test_idx = indices[train_size + val_size:]
        print('Done')

        print(f'Total nodes: {len(labels)}')
        print(f'Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}')

        return tweets_tensor, num_prop, category_prop, edge_index, edge_type, labels, train_idx, val_idx, test_idx
