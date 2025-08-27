import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch_geometric.data import Data, Batch
from torch_sparse import SparseTensor
import numpy as np
import igraph as ig
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, f1_score
from process.datahandlers import get_handler


# 构造社区子图Batch，返回PyG批次和社区子图列表
def build_community_batch(communities, node_embeddings, G, device):
    data_list = []
    ig_subgraphs = []
    for comm_id, node_names in communities.items():
        node_idxs = [G.vs.find(name=n).index for n in node_names]
        subgraph = G.subgraph(node_idxs)
        ig_subgraphs.append(subgraph)

        edges = []
        for e in subgraph.es:
            edges.append([e.source, e.target])
            edges.append([e.target, e.source])
        if len(edges) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        x = torch.tensor(np.array([node_embeddings[G.vs[i]["name"]] for i in node_idxs]), dtype=torch.float32)
        data_list.append(Data(x=x, edge_index=edge_index))

    batch = Batch.from_data_list(data_list).to(device)
    return batch, ig_subgraphs

# 用静态节点向量均值计算每个社区图表示
def compute_community_embeddings(communities, node_embeddings, G, device='cpu'):
    comm_embeddings = []
    for comm_id, node_names in communities.items():
        vectors = []
        for name in node_names:
            if name in node_embeddings:
                vectors.append(node_embeddings[name])
        if len(vectors) == 0:
            vectors = [np.zeros_like(next(iter(node_embeddings.values())))]
        comm_tensor = torch.tensor(vectors, dtype=torch.float32, device=device)
        mean_vec = comm_tensor.mean(dim=0)
        comm_embeddings.append(mean_vec)
    return torch.stack(comm_embeddings, dim=0)  # shape: (num_communities, dim)

# 计算社区间余弦相似度
def compute_cosine_similarity(graph_embeddings):
    graph_embeddings_np = graph_embeddings.detach().cpu().numpy()
    similarity_matrix = cosine_similarity(graph_embeddings_np)
    return similarity_matrix

# 根据相似度矩阵构建社区间的KNN图 SparseTensor
def build_knn_graph(similarity_matrix, k=5, device='cpu'):
    sim = similarity_matrix.copy()
    min_val = np.min(sim)
    sim[np.diag_indices_from(sim)] = min_val - 1  # 避免连接自己
    knn = kneighbors_graph(sim, n_neighbors=k, mode='connectivity', include_self=False)
    coo = knn.tocoo()

    row = torch.tensor(coo.row, dtype=torch.long, device=device)
    col = torch.tensor(coo.col, dtype=torch.long, device=device)
    value = torch.ones(len(coo.data), dtype=torch.float32, device=device)

    adj_t = SparseTensor(row=row, col=col, value=value, sparse_sizes=(sim.shape[0], sim.shape[1]))
    return adj_t

# 主训练函数
def train_model(encoder, classifier, communities, node_embeddings, G, args, ground_truth_malicious_nodes,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                save_encoder_path="encoder.pth",
                save_classifier_path="classifier.pth"):
    encoder.to(device)
    classifier.to(device)
    optimizer_e = torch.optim.Adam(encoder.parameters(), lr=args.lr)
    optimizer_c = torch.optim.Adam(classifier.parameters(), lr=args.lr)

    # Step 1: 构建社区子图 batch
    batch_data, _ = build_community_batch(communities, node_embeddings, G, device)

    # Step 2: 计算社区之间的余弦相似度图（用静态嵌入求平均）
    comm_embeds = compute_community_embeddings(communities, node_embeddings, G, device)
    sim_matrix = compute_cosine_similarity(comm_embeds)
    knn_adj_t = build_knn_graph(sim_matrix, k=args.k, device=device)

    # Step 3: 构建标签（只要社区内有恶意节点就设为1）
    y = torch.zeros(len(communities), dtype=torch.long, device=device)
    for community_id, node_list in communities.items():
        if any(node in ground_truth_malicious_nodes for node in node_list):
            y[community_id] = 1
    print(y)
    class_counts = torch.bincount(y)
    # Step 4: 构建加权损失函数
    raw_weights = 1.0 / (class_counts.float() + 1e-6)
    max_ratio = 5.0  # 最多不超过 5 倍差距
    min_w = raw_weights.min()
    clamped_weights = torch.clamp(raw_weights, max=min_w * max_ratio)
    class_weights = clamped_weights / clamped_weights.sum()
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))

    encoder.train()
    classifier.train()

    for epoch in range(1, args.epochs + 1):
        optimizer_e.zero_grad()
        optimizer_c.zero_grad()

        # 社区图嵌入
        graph_embeds = encoder(batch_data.x, batch_data.edge_index, batch_data.batch)  # shape: (num_communities, D)
        # 在 knn 图上传播社区间信息（可多层传播）
        enhanced_embeds = graph_embeds
        for _ in range(args.knn_layer):  # 添加这个参数到 args，比如设为 1 或 2
            knn_coo = knn_adj_t.to_torch_sparse_coo_tensor()
            enhanced_embeds = enhanced_embeds + torch.sparse.mm(knn_coo, enhanced_embeds)
        # 分类
        out = classifier(enhanced_embeds)

        loss = loss_fn(out, y)
        loss.backward()
        optimizer_e.step()
        optimizer_c.step()

        preds = out.argmax(dim=1).cpu().numpy()
        labels = y.cpu().numpy()

        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='macro')

        print(f"Epoch {epoch:03d}, Loss: {loss.item():.4f}, Accuracy: {acc:.4f}, F1-macro: {f1:.4f}")

    # Step 5: 保存模型
    torch.save(encoder.state_dict(), save_encoder_path)
    torch.save(classifier.state_dict(), save_classifier_path)
    print(f"模型已保存到 '{save_encoder_path}' 和 '{save_classifier_path}'")

    return encoder, classifier, knn_adj_t


