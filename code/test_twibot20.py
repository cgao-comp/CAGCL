import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from model import CAGCL
from Dataset import Twibot20
from utils import accuracy
import os

def test_model(model_path='outputs/twibot20/best_cw=0.5_temp=0.1_nsw=0.3.pt', device='cuda:1'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset = Twibot20(device=device, process=False, save=False)
    des_tensor, tweets_tensor, num_prop, category_prop, edge_index, edge_type, labels, train_idx, val_idx, test_idx = dataset.dataloader1()

    total_nodes = des_tensor.size(0)
    community_file = "outputs/twibot20/community_labels.npy"
    if os.path.exists(community_file):
        community_labels = np.load(community_file)
        num_communities = len(np.unique(community_labels))
        community_labels = torch.from_numpy(community_labels).long().to(device)
    else:
        community_labels = torch.zeros(total_nodes, dtype=torch.long, device=device)
        num_communities = 1

    community_embedding = torch.zeros(total_nodes, num_communities, device=device)
    for i in range(total_nodes):
        comm_id = community_labels[i].item()
        if 0 <= comm_id < num_communities:
            community_embedding[i, comm_id] = 1.0

    edge_community_weight = torch.zeros(edge_index.size(1), device=device)
    edge_src, edge_dst = edge_index[0], edge_index[1]
    valid_edges = (edge_src < len(community_labels)) & (edge_dst < len(community_labels))
    same_community = torch.zeros_like(edge_src, dtype=torch.bool)
    same_community[valid_edges] = community_labels[edge_src[valid_edges]] == community_labels[edge_dst[valid_edges]]
    edge_community_weight[same_community] = 1.0
    edge_community_weight[valid_edges & (~same_community)] = 0.2

    model = CAGCL(
        des_size=768,
        tweet_size=768,
        num_prop_size=6,
        cat_prop_size=11,
        community_size=num_communities,
        embedding_dimension=128,
        dropout=0.3,
        temperature=0.1,
        contrastive_weight=0.5,
        node_similarity_weight=0.3
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        test_output, _ = model(
            des_tensor, tweets_tensor, num_prop, category_prop,
            edge_index, edge_type, community_embedding, edge_community_weight,
            return_loss=False
        )
        test_pred = test_output[test_idx].max(1)[1].cpu().numpy()
        test_true = labels[test_idx].cpu().numpy()

        f1 = f1_score(test_true, test_pred)
        precision = precision_score(test_true, test_pred)
        recall = recall_score(test_true, test_pred)
        acc_test = accuracy(test_output[test_idx], labels[test_idx])

        print("\nCAGCL Final Model Test Results:")
        print(f"  Accuracy: {acc_test:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")

if __name__ == "__main__":
    test_model()
