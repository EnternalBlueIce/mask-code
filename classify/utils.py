import igraph as ig
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GINConv
import torch.nn.functional as F
from torch_scatter import segment_csr, scatter_mean
from grakel import ShortestPath
import numpy as np
from torch_geometric.nn import global_add_pool

# ------------------- 加载恶意节点 -------------------
def load_malicious_nodes(file_path):
    with open(file_path, 'r') as f:
        malicious_nodes = set(line.strip() for line in f if line.strip())
    print(f"[标签加载] 恶意节点数量: {len(malicious_nodes)}")
    return malicious_nodes

# ------------------- 给社区打标签 -------------------
def community_labeling(communities, malicious_node_names):
    labels_dict = {}
    for comm_id, nodes in communities.items():
        if isinstance(nodes[0], str):
            node_names = nodes
        else:
            node_names = [v["name"] for v in nodes]
        label = 1 if any(n in malicious_node_names for n in node_names) else 0
        labels_dict[comm_id] = label
    print(f"[社区标签] 共标注 {len(labels_dict)} 个社区")
    return labels_dict

# ------------------- igraph转PyG Data -------------------
def igraph_to_pyg_data(g):
    x = torch.tensor(g.degree(), dtype=torch.float).view(-1, 1)
    edge_index = torch.tensor(g.get_edgelist(), dtype=torch.long).t().contiguous()
    # 检查孤立节点
    num_nodes = x.size(0)
    all_nodes_in_edges = set(edge_index[0].tolist()) | set(edge_index[1].tolist())
    all_nodes = set(range(num_nodes))
    missing_nodes = all_nodes - all_nodes_in_edges
   #print(len(missing_nodes))
    if len(missing_nodes) > 0:
        print(f"警告：图中存在孤立节点（无边连接）：{missing_nodes}")
        # 给孤立节点添加自环，确保每个节点至少有一条边
        import torch_geometric.utils as pyg_utils
        edge_index, _ = pyg_utils.add_self_loops(edge_index, num_nodes=num_nodes)
        print(f"已为孤立节点添加自环边")
    else:
        # 即使没有孤立节点，也建议给所有节点添加自环（一般GNN都这么做）
        import torch_geometric.utils as pyg_utils
        edge_index, _ = pyg_utils.add_self_loops(edge_index, num_nodes=num_nodes)
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    return Data(x=x, edge_index=edge_index)

# ------------------- igraph转grakel格式 -------------------
def igraph_to_grakel(g, malicious_node_names):
    edges = g.get_edgelist()
    labels = {}
    for v in g.vs:
        node_name = v["name"]
        label = 1 if node_name in malicious_node_names else 0
        labels[v.index] = label
        v["label"] = label
    return (edges, labels)

# ------------------- 计算最短路径核 -------------------
def compute_shortest_path_kernel(subgraphs, malicious_node_names):
    gk = ShortestPath(normalize=True, with_labels=True)
    gk_graphs = [igraph_to_grakel(g, malicious_node_names) for g in subgraphs]
    kernel_matrix = gk.fit_transform(gk_graphs)
    print(f"[Step 3] 最短路径核矩阵:\n{np.round(kernel_matrix, 3)}")
    return kernel_matrix

# ------------------- 构建社区间KNN图 -------------------
def build_knn_graph(kernel_matrix, k):
    kernel_tensor = torch.tensor(kernel_matrix)
    _, knn_indices = torch.topk(kernel_tensor, k=k+1, dim=1, largest=True)
    edge_index = [[], []]
    for i in range(knn_indices.size(0)):
        for j in knn_indices[i, 1:]:
            edge_index[0].append(i)
            edge_index[1].append(j.item())
    edge_index = torch.tensor(edge_index)
    print(f"[Step 4] 社区间KNN图边数: {edge_index.size(1)}")
    return edge_index

# ------------------- GIN模型 -------------------
class GIN(torch.nn.Module):
    def __init__(self, args, use_drop=False):
        super(GIN, self).__init__()
        self.conv1 = GINConv(
            torch.nn.Sequential(
                torch.nn.Linear(args.n_feat, args.n_hidden),
                torch.nn.BatchNorm1d(args.n_hidden),
                torch.nn.ReLU(),
                torch.nn.Linear(args.n_hidden, args.n_hidden),
                torch.nn.ReLU()
            )
        )
        self.conv2 = GINConv(
            torch.nn.Sequential(
                torch.nn.Linear(args.n_hidden, args.n_hidden),
                torch.nn.BatchNorm1d(args.n_hidden),
                torch.nn.ReLU(),
                torch.nn.Linear(args.n_hidden, args.n_hidden),
                torch.nn.ReLU()
            )
        )
        self.conv3 = GINConv(
            torch.nn.Sequential(
                torch.nn.Linear(args.n_hidden, args.n_hidden),
                torch.nn.BatchNorm1d(args.n_hidden),
                torch.nn.ReLU(),
                torch.nn.Linear(args.n_hidden, args.n_hidden),
                torch.nn.ReLU()
            )
        )
        self.conv4 = GINConv(
            torch.nn.Sequential(
                torch.nn.Linear(args.n_hidden, args.n_hidden),
                torch.nn.BatchNorm1d(args.n_hidden),
                torch.nn.ReLU(),
                torch.nn.Linear(args.n_hidden, args.n_hidden),
                torch.nn.ReLU()
            )
        )
        self.conv5 = GINConv(
            torch.nn.Sequential(
                torch.nn.Linear(args.n_hidden, args.n_hidden),
                torch.nn.BatchNorm1d(args.n_hidden),
                torch.nn.ReLU(),
                torch.nn.Linear(args.n_hidden, args.n_hidden),
                torch.nn.ReLU()
            )
        )
        self.dropout = torch.nn.Dropout(p=0.2)
        self.use_drop = use_drop

    def forward(self, x, edge_index, batch=None, aggregate=False):
        x = self.conv1(x, edge_index)
        if self.use_drop:
            x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)
        x = self.conv4(x, edge_index)
        x = self.conv5(x, edge_index)

        if aggregate:
            assert batch is not None, "batch is required for aggregation"
            x = segment_csr(x, batch, reduce="sum")  # 也可以改成global_add_pool(x, batch)
        return x

# ------------------- MLP分类器 -------------------
class MLP_Classifier(torch.nn.Module):
    def __init__(self, args):
        super(MLP_Classifier, self).__init__()
        self.lin1 = torch.nn.Linear(args.n_hidden, args.n_hidden)
        self.lin2 = torch.nn.Linear(args.n_hidden, args.n_class)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)

# ------------------- 社区间传播 -------------------
class CommunityPropagate(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, dim),
            torch.nn.ReLU(),
            torch.nn.Linear(dim, dim)
        )

    def forward(self, x, edge_index):
        row, col = edge_index
        agg = scatter_mean(x[col], row, dim=0, dim_size=x.size(0))
        return self.mlp(agg)

# ------------------- 社区图编码与传播主流程 -------------------
def community_gnn_pipeline(G, communities, malicious_node_names, knn_k, args, use_drop=False):
    print(f"[Step 1] 社区数量: {len(communities)}")

    # 1. 构建社区子图列表
    subgraphs = []
    for comm_id, vertices in communities.items():
        if isinstance(vertices[0], str):
            vertex_indices = [G.vs.find(name=v).index for v in vertices]
        else:
            vertex_indices = [v.index for v in vertices]
        sg = G.subgraph(vertex_indices)
        subgraphs.append(sg)
    print(f"[Step 1] 每个社区节点数: {[sg.vcount() for sg in subgraphs]}")

    # 2. 初始化GIN模型
    gin_model = GIN(args, use_drop=use_drop)
    # 3. 对每个社区单独编码，输出社区特征向量（1个向量/社区）
    community_features_list = []
    for i, sg in enumerate(subgraphs):
        data = igraph_to_pyg_data(sg)
        batch_single = torch.zeros(data.x.size(0), dtype=torch.long)  # 所有节点归属同一个社区图
        node_emb = gin_model(data.x, data.edge_index, batch_single)  # shape: [num_nodes, hidden_dim]
        # print(f"社区{i} data.edge_index.shape: {data.edge_index.shape}")
        # print(f"社区{i} data.x.shape: {data.x.shape}")
        # print(f"社区{i} batch_single.shape: {batch_single.shape}")
        # print(f"社区{i} node_emb.shape: {node_emb.shape}")
        graph_emb = global_add_pool(node_emb, batch_single)  # shape: [1, hidden_dim]
        #print(f"社区 {i} 图表示张量 shape: {graph_emb.shape}")
        community_features_list.append(graph_emb)
    community_features = torch.cat(community_features_list, dim=0)    # shape: [社区数, hidden_dim]
    print(f"[Step 2] 每个社区图的表示维度: {community_features.shape}")  # 应该是 (22, 64) 类似的形状
    # 4. 计算社区间最短路径核矩阵
    kernel_matrix = compute_shortest_path_kernel(subgraphs, malicious_node_names)

    # 5. 构建社区间KNN图
    community_edge_index = build_knn_graph(kernel_matrix, knn_k)

    # 6. 社区间传播
    prop_model = CommunityPropagate(args.n_hidden)
    propagated_features = prop_model(community_features, community_edge_index)
    print(f"[Step 3] 传播后社区最终表示shape: {propagated_features.shape}")  # 应该还是 (社区数, hidden_dim)

    return propagated_features, community_edge_index

# ------------------- 生成标签张量 -------------------
def generate_labels_tensor(community_ids, labels_dict):
    labeled_indices = [i for i, cid in enumerate(community_ids) if cid in labels_dict]
    x_indices = torch.tensor(labeled_indices, dtype=torch.long)
    y_tensor = torch.tensor([labels_dict[cid] for cid in community_ids if cid in labels_dict], dtype=torch.long)
    print(f"[标签] 有效社区数: {len(y_tensor)}")
    return x_indices, y_tensor

# ------------------- 训练函数 -------------------
def train(model_gnn, model_classifier, optimizer_gnn, optimizer_clf,
          G, communities, malicious_node_names, knn_k, args,
          labels_dict, epochs=50, batch_size=16, device='cpu',
          output_pred_file=None):

    model_gnn.to(device)
    model_classifier.to(device)
    criterion = torch.nn.NLLLoss()

    for epoch in range(1, epochs+1):
        model_gnn.train()
        model_classifier.train()

        propagated_features, _ = community_gnn_pipeline(G, communities, malicious_node_names, knn_k, args)

        comm_ids = list(communities.keys())
        y = torch.tensor([labels_dict.get(cid, 0) for cid in comm_ids], dtype=torch.long, device=device)

        # 确保propagated_features和y样本数匹配
        assert propagated_features.size(0) == y.size(0), "社区特征数与标签数不匹配！"
        y = torch.tensor([labels_dict[cid] for cid in comm_ids], dtype=torch.long, device=device)

        print(f"Train: y.shape: {y.shape}")
        print(f"Train: propagated_features.shape: {propagated_features.shape}")
        dataset = torch.utils.data.TensorDataset(propagated_features, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        epoch_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in loader:
            optimizer_gnn.zero_grad()
            optimizer_clf.zero_grad()

            output = model_classifier(batch_x)
            loss = criterion(output, batch_y)
            #loss.backward()
            optimizer_gnn.step()
            optimizer_clf.step()

            epoch_loss += loss.item() * batch_x.size(0)
            preds = output.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

        avg_loss = epoch_loss / total
        acc = correct / total
        print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Accuracy={acc:.4f}")

    print("训练结束")

    if output_pred_file is not None:
        model_gnn.eval()
        model_classifier.eval()
        with torch.no_grad():
            propagated_features, _ = community_gnn_pipeline(G, communities, malicious_node_names, knn_k, args)
            propagated_features = propagated_features.to(device)
            output = model_classifier(propagated_features)
            preds = output.argmax(dim=1).cpu().tolist()

        with open(output_pred_file, 'w') as f:
            for p in preds:
                f.write(f"{p}\n")
        print(f"预测标签已写入: {output_pred_file}")
