import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from DANMF import DANMF
from Dataset import Cresci15
import time
import os

def to_networkx_from_edge_list(edge_index, num_nodes=None, max_nodes=5000):

    if num_nodes is None:
        num_nodes = max(edge_index[0].max().item(), edge_index[1].max().item()) + 1

    if num_nodes > max_nodes:
        print(f"Node count {num_nodes} exceeds max visualization limit {max_nodes}, applying sampling...")
        sampled_indices = np.random.choice(num_nodes, max_nodes, replace=False)
        sampled_indices_set = set(sampled_indices)
        edge_list = []
        node_map = {int(idx): i for i, idx in enumerate(sampled_indices)}

        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            if src in sampled_indices_set and dst in sampled_indices_set:
                edge_list.append((node_map[src], node_map[dst]))

        G = nx.Graph()
        G.add_nodes_from(range(len(sampled_indices)))
        G.add_edges_from(edge_list)
        return G, sampled_indices
    else:
        edge_list = [(edge_index[0, i].item(), edge_index[1, i].item()) 
                     for i in range(edge_index.size(1))]
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        G.add_edges_from(edge_list)
        return G, None


def plot_graph(G, community_labels=None, sampled_indices=None, node_size=30, title="Graph"):

    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, seed=42)

    if community_labels is not None:
        if sampled_indices is not None:
            labels_to_use = community_labels[sampled_indices]
        else:
            labels_to_use = community_labels

        for comm_id in np.unique(labels_to_use):
            comm_nodes = [i for i, label in enumerate(labels_to_use) if label == comm_id]
            nx.draw_networkx_nodes(G, pos, nodelist=comm_nodes, 
                                   node_color=f"C{comm_id}", 
                                   node_size=node_size,
                                   label=f"Community {comm_id}")
    else:
        nx.draw_networkx_nodes(G, pos, node_size=node_size)

    nx.draw_networkx_edges(G, pos, alpha=0.2, width=0.5)
    plt.title(title)
    plt.legend()
    plt.axis("off")
    plt.tight_layout()
    os.makedirs("outputs/cresci15", exist_ok=True)
    plt.savefig(f"outputs/cresci15/{title.replace(' ', '_')}.png", dpi=300)
    plt.close()
    print(f"Graph saved to outputs/cresci15/{title.replace(' ', '_')}.png")


def evaluate_communities(edge_index, num_nodes, community_labels, danmf_model):
   
    print("Evaluating community partition...")

    print("Computing silhouette score...")
    embedding = danmf_model.get_community_embedding().detach().cpu().numpy()
    if embedding.shape[0] > 10000:
        sample_indices = np.random.choice(embedding.shape[0], 10000, replace=False)
        sampled_embedding = embedding[sample_indices]
        sampled_labels = community_labels[sample_indices]
        silhouette = silhouette_score(sampled_embedding, sampled_labels) if len(np.unique(sampled_labels)) > 1 else 0
    else:
        silhouette = silhouette_score(embedding, community_labels) if len(np.unique(community_labels)) > 1 else 0

    print("Computing modularity...")
    if num_nodes > 10000:
        G, sampled_indices = to_networkx_from_edge_list(edge_index, num_nodes, max_nodes=10000)
        sampled_labels = community_labels[sampled_indices]
        communities = []
        all_nodes = set()
        for comm_id in np.unique(sampled_labels):
            comm_nodes = [i for i, label in enumerate(sampled_labels) if label == comm_id]
            communities.append(set(comm_nodes))
            all_nodes.update(comm_nodes)
        if len(all_nodes) < len(sampled_labels):
            unassigned = set(range(len(sampled_labels))) - all_nodes
            if unassigned:
                communities.append(unassigned)
    else:
        G, _ = to_networkx_from_edge_list(edge_index, num_nodes)
        communities = []
        all_nodes = set()
        for comm_id in np.unique(community_labels):
            comm_nodes = [i for i, label in enumerate(community_labels) if label == comm_id]
            communities.append(set(comm_nodes))
            all_nodes.update(comm_nodes)
        if len(all_nodes) < num_nodes:
            unassigned = set(range(num_nodes)) - all_nodes
            if unassigned:
                communities.append(unassigned)

    print(f"Computing modularity over {len(communities)} communities...")
    try:
        modularity = nx.algorithms.community.modularity(G, communities)
    except nx.exception.NetworkXError as e:
        print(f"Error computing modularity: {e}")
        modularity = 0

    print("Computing intra- and inter-community density...")
    intra_density = 0
    inter_density = 0

    for comm in communities:
        if len(comm) <= 1:
            continue
        subgraph = G.subgraph(comm)
        intra_density += nx.density(subgraph)

    if communities:
        intra_density /= len(communities)

    max_pairs = 1000
    pairs_count = 0

    for i, comm1 in enumerate(communities):
        for j, comm2 in enumerate(communities):
            if i >= j or pairs_count >= max_pairs:
                continue
            max_nodes_per_comm = 100
            comm1_sample = list(comm1)[:max_nodes_per_comm] if len(comm1) > max_nodes_per_comm else comm1
            comm2_sample = list(comm2)[:max_nodes_per_comm] if len(comm2) > max_nodes_per_comm else comm2
            cross_edges = sum(1 for u in comm1_sample for v in comm2_sample if G.has_edge(u, v))
            possible_edges = len(comm1_sample) * len(comm2_sample)
            if possible_edges > 0:
                inter_density += cross_edges / possible_edges
            pairs_count += 1

    total_inter_pairs = min(max_pairs, len(communities) * (len(communities) - 1) / 2)
    if total_inter_pairs > 0:
        inter_density /= total_inter_pairs

    return {
        "silhouette": silhouette,
        "modularity": modularity,
        "intra_density": intra_density,
        "inter_density": inter_density,
        "separation": intra_density - inter_density
    }


def save_community_data(community_labels, labels, train_idx, val_idx, test_idx):
    print("Saving community data...")
    community_data = []

    if isinstance(labels, torch.Tensor):
        labels_np = labels.cpu().numpy()
    else:
        labels_np = labels

    train_set = set([int(i) for i in train_idx])
    val_set = set([int(i) for i in val_idx])
    test_set = set([int(i) for i in test_idx])
    labeled_indices = train_set.union(val_set).union(test_set)

    unique_communities = np.unique(community_labels)
    for comm_id in unique_communities:
        indices = np.where(community_labels == comm_id)[0]
        indices_set = set(indices)
        print(f"Processing community {comm_id}, total nodes: {len(indices)}")

        comm_train_idx = [int(idx) for idx in indices if idx in train_set]
        comm_val_idx = [int(idx) for idx in indices if idx in val_set]
        comm_test_idx = [int(idx) for idx in indices if idx in test_set]

        valid_indices = [idx for idx in indices if idx < len(labels_np)]
        if valid_indices:
            comm_labels = labels_np[valid_indices]
            pos_count = np.sum(comm_labels == 1)
            neg_count = np.sum(comm_labels == 0)
            pos_ratio = float(pos_count / len(valid_indices)) if len(valid_indices) > 0 else 0
        else:
            pos_count = 0
            neg_count = 0
            pos_ratio = 0

        community_info = {
            "community_id": int(comm_id),
            "size": int(len(indices)),
            "labeled_size": len(valid_indices),
            "train_size": len(comm_train_idx),
            "val_size": len(comm_val_idx),
            "test_size": len(comm_test_idx),
            "positive_count": int(pos_count),
            "negative_count": int(neg_count),
            "positive_ratio": pos_ratio
        }

        if len(indices) < 10000:
            community_info["indices"] = indices.tolist()
            community_info["train_indices"] = comm_train_idx
            community_info["val_indices"] = comm_val_idx
            community_info["test_indices"] = comm_test_idx

        community_data.append(community_info)

    import json
    os.makedirs("outputs/cresci15", exist_ok=True)

    with open("outputs/cresci15/community_data.json", "w") as f:
        json.dump(community_data, f, indent=4)

    print("Community data saved to outputs/cresci15/community_data.json")


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.cuda.set_device(0)
    else:
        device = torch.device('cpu')
        print("Warning: No GPU detected. Running on CPU.")
    print(f"Using device: {device}")

    os.makedirs("outputs/cresci15", exist_ok=True)
    start_time = time.time()

    print("Loading Cresci15 dataset...")
    dataset = Cresci15(device=device, process=False, save=False)
    des_tensor, tweets_tensor, num_prop, category_prop, edge_index, edge_type, labels, train_idx, val_idx, test_idx = dataset.dataloader1()

    num_nodes = des_tensor.size(0)
    print(f"Dataset info:\n  Nodes: {num_nodes}\n  Edges: {edge_index.size(1)}\n  Train: {len(train_idx)}\n  Val: {len(val_idx)}\n  Test: {len(test_idx)}")

    labels_np = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels
    labels_len = len(labels_np)
    print(f"  Label size: {labels_len}")

    model_path = "outputs/cresci15/danmf_model.pt"
    layer_sizes = [128, 64, 2]

    if os.path.exists(model_path):
        print(f"Loading saved model: {model_path}")
        danmf_model = DANMF(num_nodes=num_nodes, layers=layer_sizes, pre_iterations=25, iterations=25, device=device)
        danmf_model.load_model(model_path)
    else:
        print("\nInitializing DANMF model...")
        danmf_model = DANMF(num_nodes=num_nodes, layers=layer_sizes, pre_iterations=25, iterations=25, device=device)
        print("\nTraining DANMF model...")
        danmf_model.fit(edge_index)
        danmf_model.save_model(model_path)
        print(f"Model saved to: {model_path}")

    print("\nGetting community labels...")
    community_labels = danmf_model.get_communities(n_clusters=2)
    np.save("outputs/cresci15/community_labels.npy", community_labels)
    print("Community labels saved.")

    print("\nSplitting into subgraphs...")
    subgraph_edge_indices, _ = danmf_model.split_graph(edge_index, n_subgraphs=2)

    included_nodes = set()
    for subgraph_edge_index in subgraph_edge_indices:
        if subgraph_edge_index.numel() > 0:
            included_nodes.update(subgraph_edge_index.flatten().tolist())

    isolated_nodes = [i for i in range(num_nodes) if i not in included_nodes]
    print(f"Isolated nodes found: {len(isolated_nodes)}")

    for node_idx in isolated_nodes:
        community_id = community_labels[node_idx]
        self_loop = torch.tensor([[node_idx], [node_idx]], device=edge_index.device)
        if subgraph_edge_indices[community_id].numel() > 0:
            self_loop = self_loop.to(subgraph_edge_indices[community_id].device)
            subgraph_edge_indices[community_id] = torch.cat([subgraph_edge_indices[community_id], self_loop], dim=1)
        else:
            subgraph_edge_indices[community_id] = self_loop

    total_nodes = 0
    for i, subgraph_edge_index in enumerate(subgraph_edge_indices):
        if subgraph_edge_index.numel() > 0:
            nodes_set = set(subgraph_edge_index.flatten().tolist())
            total_nodes += len(nodes_set)
            print(f"Subgraph {i}: Nodes={len(nodes_set)}, Edges={subgraph_edge_index.size(1)}")
        else:
            print(f"Subgraph {i} is empty")

    if total_nodes != num_nodes:
        print(f"Warning: Mismatch in total nodes. Found: {total_nodes}, Expected: {num_nodes}")
    else:
        print("All nodes assigned to subgraphs")

    print("\nEvaluating communities...")
    metrics = evaluate_communities(edge_index, num_nodes, community_labels, danmf_model)
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

    import json
    with open("outputs/cresci15/metrics.json", "w") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=4)

    save_community_data(community_labels, labels, train_idx, val_idx, test_idx)

    print("\nVisualizing graphs...")
    # Visualization logic skipped here for brevity

    for i, subgraph_edge_index in enumerate(subgraph_edge_indices):
        if subgraph_edge_index.numel() > 0:
            np.save(f"outputs/cresci15/subgraph_{i}_edge_index.npy", subgraph_edge_index.cpu().numpy())
            print(f"Subgraph {i} edges saved")

    elapsed_time = time.time() - start_time
    h, rem = divmod(elapsed_time, 3600)
    m, s = divmod(rem, 60)
    print(f"\nDANMF complete. Runtime: {int(h)}h {int(m)}m {s:.2f}s")

    print("\nCommunity statistics:")
    for i, comm_id in enumerate(np.unique(community_labels)):
        indices = np.where(community_labels == comm_id)[0]
        indices_set = set(indices)
        labeled_indices = [idx for idx in indices if idx < labels_len]

        if labeled_indices:
            comm_labels = labels_np[labeled_indices]
            pos_ratio = np.mean(comm_labels == 1) if len(labeled_indices) > 0 else 0
        else:
            pos_ratio = "Unknown"

        train_in_comm = len(indices_set & set([int(i) for i in train_idx]))
        val_in_comm = len(indices_set & set([int(i) for i in val_idx]))
        test_in_comm = len(indices_set & set([int(i) for i in test_idx]))

        print(f"  Community {comm_id} (Subgraph {i}):")
        print(f"    Total nodes: {len(indices)}")
        print(f"    Labeled nodes: {len(labeled_indices)}")
        if isinstance(pos_ratio, str):
            print(f"    Positive ratio: {pos_ratio}")
        else:
            print(f"    Positive ratio: {pos_ratio:.4f}")
            print(f"    Train nodes: {train_in_comm}")
            print(f"    Val nodes: {val_in_comm}")
            print(f"    Test nodes: {test_in_comm}")
