import torch
import torch.nn.functional as F
from torch_geometric.utils import from_networkx
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from process.datahandlers import get_handler
from process.partition import detect_communities
from types import SimpleNamespace
from encoder import GIN, MLP_Classifier
import networkx as nx


def igraph_to_networkx(ig_g):
    nx_g = nx.DiGraph() if ig_g.is_directed() else nx.Graph()
    for v in ig_g.vs:
        name = v['name'] if 'name' in v.attributes() else str(v.index)
        nx_g.add_node(name, **v.attributes(), node_id=name)
    for e in ig_g.es:
        source = ig_g.vs[e.source]['name']
        target = ig_g.vs[e.target]['name']
        nx_g.add_edge(source, target, **e.attributes())
    return nx_g


def test_community_node(communities, ground_truths, predictions):
    y_true = []
    y_pred = []
    tp, fp, tn, fn = 0, 0, 0, 0
    if isinstance(communities, dict):
        community_iter = communities.items()
    else:
        community_iter = enumerate(communities)

    for _, community in community_iter:
        for node_name in community:
            pred = int(node_name in predictions)
            label = int(node_name in ground_truths)
            y_true.append(label)
            y_pred.append(pred)
            if label == 1 and pred == 1:
                tp += 1
            elif label == 0 and pred == 1:
                fp += 1
            elif label == 1 and pred == 0:
                fn += 1
            elif label == 0 and pred == 0:
                tn += 1

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_pred) if len(set(y_true)) > 1 else None
    fpr = fp / (fp + tn + 1e-10)
    attack_coverage = tp / len(ground_truths)
    workload_reduction = tp / (tp + fp + 1e-10)

    print("\n📊评估结果：")
    print(f"✅ Accuracy:  {acc:.4f}")
    print(f"✅ Precision: {prec:.4f}")
    print(f"✅ Recall:    {rec:.4f}")
    print(f"✅ F1 Score:  {f1:.4f}")
    print(f"✅ AUC:       {auc:.4f}" if auc else "⚠️ AUC 无法计算")
    print(f"✅ FPR:       {fpr:.4f}")
    print(f"✅ TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
    print(f"✅ attack_coverage: {attack_coverage:.4f}, workload_reduction: {workload_reduction:.4f}")
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
        "fpr": fpr,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn
    }


def load_model(model_class, model_path, args, device='cpu'):
    model = model_class(args)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # === 参数配置 ===
    args = SimpleNamespace(
        n_feat=30,
        n_hidden=64,
        n_class=2,
        lr=0.001
    )

    # === 加载数据 ===
    data_handler = get_handler("atlas", False)
    data_handler.load()
    features, edges, mapp, relations, G = data_handler.build_graph()
    communities = detect_communities(G)

    # === 特征映射 ===
    features_dict = {name: features[idx] for idx, name in enumerate(mapp)}

    # === 加载模型 ===
    encoder = load_model(GIN, r"D:\数据集分析\Flash-IDS-main\process\classify\encoder.pth", args, device)
    classifier = load_model(MLP_Classifier, r"D:\数据集分析\Flash-IDS-main\process\classify\classifier.pth", args, device)

    # === 转为 NetworkX 图 ===
    G_nx = igraph_to_networkx(G)

    # === 节点级预测 ===
    predicted_node_names = []

    if isinstance(communities, dict):
        community_iter = communities.items()
    else:
        community_iter = enumerate(communities)

    for community_id, node_list in communities.items():
        subgraph = G_nx.subgraph(node_list).copy()

        # 取原始节点名称作为 node_names
        node_names = list(subgraph.nodes())

        # 将节点名作为 node_id 属性加入 NetworkX 节点属性中
        for node in node_names:
            subgraph.nodes[node]['node_id'] = node

        # 转换为 PyG Data 对象
        data = from_networkx(subgraph)

        # 为当前子图中每个节点构造特征向量列表
        x_list = []
        valid_node_names = []
        for n in node_names:
            if n in features_dict:
                x_list.append(features_dict[n])
                valid_node_names.append(n)
            else:
                print(f"⚠️ 节点缺失特征，跳过: {n}")

        if not x_list:
            print(f"❌ 社区 {community_id} 中无有效特征节点，跳过")
            continue
        print(x_list)
        # 构造特征张量和 batch
        x = torch.tensor(x_list, dtype=torch.float32)
        data.x = x
        data.batch = torch.zeros(len(valid_node_names), dtype=torch.long)

        with torch.no_grad():
            node_embeds = encoder(data.x.to(device), data.edge_index.to(device), data.batch.to(device))
            logits = classifier(node_embeds)
            preds = logits.argmax(dim=1).cpu().numpy()

        for i, label in enumerate(preds):
            if label == 1:
                predicted_node_names.append(data.node_id[i])

    # === 评估 ===
    result = test_community_node(communities, data_handler.all_labels, predicted_node_names)
