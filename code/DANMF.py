import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import networkx as nx
from sklearn.cluster import KMeans
import os
from tqdm import tqdm

class DANMF(object):
    def __init__(self, num_nodes, layers, pre_iterations=100, iterations=100, 
                 lamb=0.01, device='cpu', seed=42):
        self.num_nodes = num_nodes
        self.layers = layers
        self.pre_iterations = pre_iterations
        self.iterations = iterations
        self.lamb = lamb
        self.device = device
        self.seed = seed
        self._setup_model()
        
    def _setup_model(self):
        torch.manual_seed(self.seed)
        self.encoder_weight_list = []
        self.encoder_bias_list = []
        self.decoder_weight_list = []
        self.decoder_bias_list = []
        self.encoder_acts = []
        self.decoder_acts = []
        
    def _setup_layer_parameters(self, i, input_size):
        encoder_weight = torch.abs(torch.randn(input_size, self.layers[i], device=self.device))
        encoder_bias = torch.abs(torch.randn(self.layers[i], device=self.device))
        decoder_weight = torch.abs(torch.randn(self.layers[i], input_size, device=self.device)) 
        decoder_bias = torch.abs(torch.randn(input_size, device=self.device))
        encoder_weight = nn.Parameter(encoder_weight, requires_grad=True)
        encoder_bias = nn.Parameter(encoder_bias, requires_grad=True)
        decoder_weight = nn.Parameter(decoder_weight, requires_grad=True)
        decoder_bias = nn.Parameter(decoder_bias, requires_grad=True)
        self.encoder_weight_list.append(encoder_weight)
        self.encoder_bias_list.append(encoder_bias)
        self.decoder_weight_list.append(decoder_weight)
        self.decoder_bias_list.append(decoder_bias)
    
    def _create_sparse_features(self, edge_index):
        edge_index_np = edge_index.cpu().numpy()
        degrees = np.zeros(self.num_nodes)
        for i in range(edge_index.size(1)):
            src = edge_index_np[0, i]
            dst = edge_index_np[1, i]
            degrees[src] += 1
            degrees[dst] += 1
        degrees = degrees / (np.sum(degrees) + 1e-10)
        node_features = torch.zeros((self.num_nodes, self.layers[0]), device=self.device)
        node_features[:, 0] = torch.FloatTensor(degrees).to(self.device)
        if self.layers[0] > 1:
            random_features = torch.abs(torch.randn(self.num_nodes, self.layers[0]-1, device=self.device))
            node_features[:, 1:] = random_features
        return node_features
    
    def _graph_convolution(self, features, edge_index):
        src, dst = edge_index
        result = torch.zeros_like(features)
        for i in range(edge_index.size(1)):
            result[dst[i]] += features[src[i]]
        degrees = torch.zeros(self.num_nodes, device=self.device)
        for i in range(edge_index.size(1)):
            degrees[dst[i]] += 1
        degrees = torch.clamp(degrees, min=1)
        result = result / degrees.unsqueeze(1)
        return result
    
    def _forward_encoder(self, x, layer_idx):
        x = torch.mm(x, self.encoder_weight_list[layer_idx])
        x = x + self.encoder_bias_list[layer_idx]
        return F.relu(x)
    
    def _forward_decoder(self, x, layer_idx):
        x = torch.mm(x, self.decoder_weight_list[layer_idx])
        x = x + self.decoder_bias_list[layer_idx]
        return F.relu(x)
    
    def _pretrain_layer(self, layer_input, layer_idx):
        if layer_idx == 0:
            input_size = self.layers[0]
        else:
            input_size = self.layers[layer_idx-1]
        self._setup_layer_parameters(layer_idx, input_size)
        layer_input_copy = layer_input.detach().clone().requires_grad_(True)
        for iteration in range(self.pre_iterations):
            encoder_output = self._forward_encoder(layer_input_copy, layer_idx)
            decoder_output = self._forward_decoder(encoder_output, layer_idx)
            loss = torch.mean((layer_input_copy - decoder_output) ** 2)
            reg_loss = 0
            for param in [self.encoder_weight_list[layer_idx], self.encoder_bias_list[layer_idx],
                         self.decoder_weight_list[layer_idx], self.decoder_bias_list[layer_idx]]:
                reg_loss += torch.sum(param ** 2)
            loss = loss + self.lamb * reg_loss
            self._update_weights(loss, layer_idx)
        with torch.no_grad():
            return self._forward_encoder(layer_input, layer_idx)
    
    def _update_weights(self, loss, layer_idx):
        lr = 0.01
        optimizer = torch.optim.Adam([
            self.encoder_weight_list[layer_idx],
            self.encoder_bias_list[layer_idx],
            self.decoder_weight_list[layer_idx],
            self.decoder_bias_list[layer_idx]
        ], lr=lr)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            self.encoder_weight_list[layer_idx].data.clamp_(min=0)
            self.encoder_bias_list[layer_idx].data.clamp_(min=0)
            self.decoder_weight_list[layer_idx].data.clamp_(min=0)
            self.decoder_bias_list[layer_idx].data.clamp_(min=0)
    
    def fit(self, edge_index):
        node_features = self._create_sparse_features(edge_index)
        self.encoder_acts = []
        encoded_input = node_features
        for i in range(len(self.layers)):
            if i > 0:
                with torch.no_grad():
                    conv_features = self._graph_convolution(encoded_input, edge_index)
                    encoded_input = conv_features
            encoded_output = self._pretrain_layer(encoded_input, i)
            self.encoder_acts.append(encoded_output)
            encoded_input = encoded_output.detach()
        total_loss = 0.0
        progress_bar = tqdm(range(self.iterations), desc="Training")
        for iteration in progress_bar:
            node_features_copy = node_features.detach().clone().requires_grad_(True)
            encoded_input = node_features_copy
            for i in range(len(self.layers)):
                if i > 0:
                    encoded_input = self._graph_convolution(encoded_input, edge_index)
                encoded_input = self._forward_encoder(encoded_input, i)
            decoded_output = encoded_input
            for i in range(len(self.layers)-1, -1, -1):
                decoded_output = self._forward_decoder(decoded_output, i)
            loss = torch.mean((node_features_copy - decoded_output) ** 2)
            reg_loss = 0
            for i in range(len(self.layers)):
                for param in [self.encoder_weight_list[i], self.encoder_bias_list[i],
                             self.decoder_weight_list[i], self.decoder_bias_list[i]]:
                    reg_loss += torch.sum(param ** 2)
            loss = loss + self.lamb * reg_loss
            all_params = []
            for i in range(len(self.layers)):
                all_params.extend([
                    self.encoder_weight_list[i],
                    self.encoder_bias_list[i],
                    self.decoder_weight_list[i],
                    self.decoder_bias_list[i]
                ])
            optimizer = torch.optim.Adam(all_params, lr=0.01)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                for i in range(len(self.layers)):
                    self.encoder_weight_list[i].data.clamp_(min=0)
                    self.encoder_bias_list[i].data.clamp_(min=0)
                    self.decoder_weight_list[i].data.clamp_(min=0)
                    self.decoder_bias_list[i].data.clamp_(min=0)
            current_loss = loss.item()
            total_loss += current_loss
            avg_loss = total_loss / (iteration + 1)
            progress_bar.set_description(f"Training [Loss: {current_loss:.6f}, Avg: {avg_loss:.6f}]")
            progress_bar.refresh()
        progress_bar.close()
        return self
    
    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        model_data = {
            'num_nodes': self.num_nodes,
            'layers': self.layers,
            'pre_iterations': self.pre_iterations,
            'iterations': self.iterations,
            'lamb': self.lamb,
            'seed': self.seed,
            'encoder_weights': [w.data.cpu() for w in self.encoder_weight_list],
            'encoder_biases': [b.data.cpu() for b in self.encoder_bias_list],
            'decoder_weights': [w.data.cpu() for w in self.decoder_weight_list],
            'decoder_biases': [b.data.cpu() for b in self.decoder_bias_list],
            'encoder_acts': [act.cpu() for act in self.encoder_acts] if self.encoder_acts else []
        }
        torch.save(model_data, path)
    
    def load_model(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file {path} not found")
        model_data = torch.load(path, map_location=self.device)
        self.num_nodes = model_data['num_nodes']
        self.layers = model_data['layers']
        self.pre_iterations = model_data['pre_iterations']
        self.iterations = model_data['iterations']
        self.lamb = model_data['lamb']
        self.seed = model_data['seed']
        self.encoder_weight_list = []
        self.encoder_bias_list = []
        self.decoder_weight_list = []
        self.decoder_bias_list = []
        for i in range(len(self.layers)):
            encoder_weight = nn.Parameter(model_data['encoder_weights'][i].to(self.device), requires_grad=True)
            encoder_bias = nn.Parameter(model_data['encoder_biases'][i].to(self.device), requires_grad=True)
            decoder_weight = nn.Parameter(model_data['decoder_weights'][i].to(self.device), requires_grad=True)
            decoder_bias = nn.Parameter(model_data['decoder_biases'][i].to(self.device), requires_grad=True)
            self.encoder_weight_list.append(encoder_weight)
            self.encoder_bias_list.append(encoder_bias)
            self.decoder_weight_list.append(decoder_weight)
            self.decoder_bias_list.append(decoder_bias)
        self.encoder_acts = [act.to(self.device) for act in model_data['encoder_acts']]
    
    def get_community_embedding(self):
        return self.encoder_acts[-1]
    
    def get_communities(self, n_clusters=None):
        if n_clusters is None:
            n_clusters = self.layers[-1]
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.seed)
        community_labels = kmeans.fit_predict(self.get_community_embedding().detach().cpu().numpy())
        return community_labels
    
    def split_graph(self, edge_index, n_subgraphs=2):
        community_labels = self.get_communities(n_clusters=n_subgraphs)
        node_to_community = {i: community_labels[i] for i in range(len(community_labels))}
        community_node_maps = {}
        for comm_id in range(n_subgraphs):
            comm_nodes = [i for i, label in enumerate(community_labels) if label == comm_id]
            community_node_maps[comm_id] = {old_idx: new_idx for new_idx, old_idx in enumerate(comm_nodes)}
        subgraph_edge_indices = []
        edge_index_np = edge_index.cpu().numpy()
        for comm_id in range(n_subgraphs):
            node_map = community_node_maps[comm_id]
            subgraph_edges = []
            for i in range(edge_index.shape[1]):
                src, dst = int(edge_index_np[0, i]), int(edge_index_np[1, i])
                if src in node_map and dst in node_map:
                    new_src, new_dst = node_map[src], node_map[dst]
                    subgraph_edges.append([new_src, new_dst])
            if subgraph_edges:
                subgraph_edge_index = torch.tensor(subgraph_edges, dtype=torch.long).t()
                subgraph_edge_indices.append(subgraph_edge_index)
            else:
                subgraph_edge_indices.append(torch.zeros((2, 0), dtype=torch.long))
        return subgraph_edge_indices, community_labels
