import torch
from torch import nn
from torch_geometric.nn import RGCNConv, BatchNorm
import torch.nn.functional as F

class CAGCL(nn.Module):
    def __init__(self, tweet_size=768, num_prop_size=10, cat_prop_size=10, 
                 community_size=2, embedding_dimension=128, dropout=0.3, 
                 temperature=0.1, contrastive_weight=0.3, node_similarity_weight=0.2):
        super(CAGCL, self).__init__()
        self.dropout = dropout
        self.temperature = temperature
        self.contrastive_weight = contrastive_weight
        self.node_similarity_weight = node_similarity_weight

        tweet_dim = int(embedding_dimension * 0.25)
        num_prop_dim = int(embedding_dimension * 0.25)
        cat_prop_dim = int(embedding_dimension * 0.25)
        comm_dim = int(embedding_dimension * 0.25)

        self.linear_relu_tweet = nn.Sequential(
            nn.Linear(tweet_size, tweet_dim),
            nn.BatchNorm1d(tweet_dim),
            nn.LeakyReLU()
        )
        self.linear_relu_num_prop = nn.Sequential(
            nn.Linear(num_prop_size, num_prop_dim),
            nn.BatchNorm1d(num_prop_dim),
            nn.LeakyReLU()
        )
        self.linear_relu_cat_prop = nn.Sequential(
            nn.Linear(cat_prop_size, cat_prop_dim),
            nn.BatchNorm1d(cat_prop_dim),
            nn.LeakyReLU()
        )
        self.linear_relu_community = nn.Sequential(
            nn.Linear(community_size, comm_dim),
            nn.BatchNorm1d(comm_dim),
            nn.LeakyReLU()
        )
        self.linear_relu_input = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.BatchNorm1d(embedding_dimension),
            nn.LeakyReLU(),
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.BatchNorm1d(embedding_dimension),
            nn.LeakyReLU()
        )
        self.rgcn1 = RGCNConv(embedding_dimension, embedding_dimension, num_relations=2)
        self.bn1 = BatchNorm(embedding_dimension)
        self.rgcn2 = RGCNConv(embedding_dimension, embedding_dimension, num_relations=2)
        self.bn2 = BatchNorm(embedding_dimension)
        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.BatchNorm1d(embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2 = nn.Linear(embedding_dimension, 2)
        self.projection_head = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.BatchNorm1d(embedding_dimension),
            nn.ReLU(),
            nn.Linear(embedding_dimension, embedding_dimension // 2),
            nn.BatchNorm1d(embedding_dimension // 2)
        )

        print(f"Feature dimensions: tweet={tweet_dim}, numerical={num_prop_dim}, categorical={cat_prop_dim}, community={comm_dim}")
        print(f"Total embedding dimension: {tweet_dim + num_prop_dim + cat_prop_dim + comm_dim}")

    def forward(self, tweet, num_prop, cat_prop, edge_index, edge_type, 
                community_embedding, edge_community_weight, return_loss=False):
        t = self.linear_relu_tweet(tweet)
        n = self.linear_relu_num_prop(num_prop)
        c = self.linear_relu_cat_prop(cat_prop)
        comm = self.linear_relu_community(community_embedding)
        x = torch.cat((t, n, c, comm), dim=1)
        x = self.linear_relu_input(x)

        x1 = self.rgcn1(x, edge_index, edge_type)
        x1 = self.bn1(x1)

        edge_src, edge_dst = edge_index[0], edge_index[1]
        mask = edge_community_weight > 0.5
        if mask.sum() > 0:
            src_nodes = edge_src[mask]
            dst_nodes = edge_dst[mask]
            weights = edge_community_weight[mask].unsqueeze(1)
            enhancement = torch.zeros_like(x1)
            enhancement.index_add_(0, dst_nodes, weights * x[src_nodes] * 0.1)
            x1 = x1 + enhancement

        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        x2 = self.rgcn2(x1, edge_index, edge_type)
        x2 = self.bn2(x2)
        features = self.linear_relu_output1(x2)
        logits = self.linear_output2(features)
        output = F.log_softmax(logits, dim=1)

        if return_loss:
            projection = self.projection_head(features)
            projection = F.normalize(projection, dim=1)
            contrastive_loss = self.compute_contrastive_loss(projection, community_embedding)
            return output, contrastive_loss

        return output, 0

    def compute_contrastive_loss(self, features, community_embedding):
        batch_size = features.size(0)
        sample_size = min(5000, batch_size) 
        
        if batch_size > sample_size:
            step = max(1, batch_size // sample_size)
            indices = torch.arange(0, batch_size, step, device=features.device)[:sample_size]
            sampled_features = features[indices]
            sampled_community = community_embedding[indices]
        else:
            sampled_features = features
            sampled_community = community_embedding
        
        similarity = torch.matmul(sampled_features, sampled_features.t()) / self.temperature
        community_sim = torch.matmul(sampled_community, sampled_community.t())
        community_mask = (community_sim > 0).float()
        N = sampled_features.shape[0]
        mask_self = torch.eye(N, device=features.device)
        community_mask = community_mask * (1 - mask_self)
        pos_per_sample = torch.clamp(community_mask.sum(dim=1), min=1.0)
        pos_similarity = torch.sum(similarity * community_mask, dim=1) / pos_per_sample
        
        neg_mask = 1.0 - community_mask - mask_self
        neg_similarity = torch.exp(similarity) * neg_mask
        neg_sum = neg_similarity.sum(dim=1)
        
        loss = -pos_similarity + torch.log(neg_sum + 1e-10)
        l2_reg = torch.norm(features, p=2, dim=1).mean()
        node_sim_loss = -torch.mean(similarity * community_mask)
        total_loss = loss.mean() + self.node_similarity_weight * node_sim_loss + 0.001 * l2_reg
        
        return self.contrastive_weight * total_loss
