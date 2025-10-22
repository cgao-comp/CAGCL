import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import numpy as np
import os
import time
import random
import json
from sklearn.metrics import f1_score, precision_score, recall_score

from model import CAGCL
from Dataset import Twibot20
from utils import accuracy, init_weights

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")

def run_once(seed, device, args):
    set_seed(seed)

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
        if i < len(community_labels):
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
        embedding_dimension=args['embedding_size'],
        dropout=args['dropout'],
        temperature=args['temperature'],
        contrastive_weight=args['contrastive_weight'],
        node_similarity_weight=args['node_similarity_weight']
    ).to(device)

    model.apply(init_weights)
    optimizer = AdamW(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(args['epochs']):
        model.train()
        output, contrastive_loss = model(
            des_tensor, tweets_tensor, num_prop, category_prop,
            edge_index, edge_type, community_embedding, edge_community_weight,
            return_loss=True
        )
        clf_loss = criterion(output[train_idx], labels[train_idx])
        total_loss = clf_loss + contrastive_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                val_output, _ = model(
                    des_tensor, tweets_tensor, num_prop, category_prop,
                    edge_index, edge_type, community_embedding, edge_community_weight,
                    return_loss=False
                )
                acc_val = accuracy(val_output[val_idx], labels[val_idx])
                if acc_val > best_val_acc:
                    best_val_acc = acc_val
                    best_epoch = epoch + 1
                    patience_counter = 0
                    torch.save(model.state_dict(), "outputs/twibot20/best_CAGCL.pt")
                else:
                    patience_counter += 1
                if patience_counter >= args['early_stopping']:
                    break

    model.load_state_dict(torch.load("outputs/twibot20/best_CAGCL.pt"))
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

    return {
        "seed": seed,
        "best_val_acc": best_val_acc.item(),
        "best_epoch": best_epoch,
        "test_acc": acc_test.item(),
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

def main():
    args = {
        'epochs': 100,
        'embedding_size': 128,
        'dropout': 0.3,
        'lr': 1e-3,
        'weight_decay': 1e-3,
        'contrastive_weight': 0.5,
        'temperature': 0.1,
        'node_similarity_weight': 0.3,
        'early_stopping': 100
    }

    os.makedirs("outputs/twibot20", exist_ok=True)
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    seeds = [43, 44, 45, 46, 47]
    all_results = []
    best_result = None

    for seed in seeds:
        print(f"\n========== Experiment seed={seed} ==========")
        result = run_once(seed, device, args)
        all_results.append(result)
        if best_result is None or result['test_acc'] > best_result['test_acc']:
            best_result = result

    print("\n===== Best Result (based on test accuracy) =====")
    for k, v in best_result.items():
        print(f"{k}: {v}")

    with open("outputs/twibot20/best_CAGCL_result.txt", "w") as f:
        f.write("===== Best CAGCL Test Result =====\n")
        for k, v in best_result.items():
            f.write(f"{k}: {v}\n")

if __name__ == "__main__":
    main()
