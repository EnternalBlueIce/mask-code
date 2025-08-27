import math
import time
import numpy as np
import copy
import torch
from process.match.dataset import FixedGraphEditDistanceDataset
from process.match.evaluation import compute_similarity, auc, compute_metrics, eval_all_metrics
from process.match.graphembeddingnetwork import GraphEncoder, GraphAggregator
from process.match.graphmatchingnetwork import GraphMatchingNet
from torch_geometric.nn import GNNExplainer

def find_important_nodes_and_edges(graph_edge_mask, edge_index):
    # **计算节点的重要性（累加与该节点相连的边的分数）**
    num_nodes = edge_index.max().item() + 1  # 计算最大节点索引，确定节点数量
    node_importance = torch.zeros(num_nodes)
    for idx, importance in enumerate(graph_edge_mask.cpu().detach().numpy()):
        src, dst = edge_index[:, idx]  # 获取边的两个端点
        node_importance[src] += importance  # 源节点
        node_importance[dst] += importance  # 目标节点

    # **按重要性排序**
    sorted_nodes = sorted(enumerate(node_importance.numpy()), key=lambda x: x[1], reverse=True)
    sorted_edges = sorted(enumerate(graph_edge_mask.cpu().detach().numpy()), key=lambda x: x[1], reverse=True)

    # **打印最重要的前 5 个节点**
    print(" 最重要的节点（前 5）：")
    for idx, importance in sorted_nodes[:5]:
        print(f"节点 {idx} → 重要性: {importance:.4f}")

    # **打印最重要的前 5 条边**
    print(" 最重要的边（前 5）：")
    for idx, importance in sorted_edges[:5]:
        print(f"边 {edge_index[:, idx].tolist()} → 重要性: {importance:.4f}")


def get_default_config():
    """The default configs."""
    model_type = 'matching'
    # Set to `embedding` to use the graph embedding net.
    node_state_dim = 32
    edge_state_dim = 16
    graph_rep_dim = 128
    graph_embedding_net_config = dict(
        node_state_dim=node_state_dim,
        edge_state_dim=edge_state_dim,
        edge_hidden_sizes=[node_state_dim * 2, node_state_dim * 2],
        node_hidden_sizes=[node_state_dim * 2],
        n_prop_layers=5,
        # set to False to not share parameters across message passing layers
        share_prop_params=True,
        # initialize message MLP with small parameter weights to prevent
        # aggregated message vectors blowing up, alternatively we could also use
        # e.g. layer normalization to keep the scale of these under control.
        edge_net_init_scale=0.1,
        # other types of update like `mlp` and `residual` can also be used here. gru
        node_update_type='gru',
        # set to False if your graph already contains edges in both directions.
        use_reverse_direction=True,
        # set to True if your graph is directed
        reverse_dir_param_different=False,
        # we didn't use layer norm in our experiments but sometimes this can help.
        layer_norm=False,
        # set to `embedding` to use the graph embedding net.
        prop_type=model_type)
    graph_matching_net_config = graph_embedding_net_config.copy()
    graph_matching_net_config['similarity'] = 'dotproduct'  # other: euclidean, cosine
    return dict(
        encoder=dict(
            node_hidden_sizes=[node_state_dim],
            node_feature_dim=1,
            edge_hidden_sizes=[edge_state_dim]),
        aggregator=dict(
            node_hidden_sizes=[graph_rep_dim],
            graph_transform_sizes=[graph_rep_dim],
            input_size=[node_state_dim],
            gated=True,
            aggregation_type='sum'),
        graph_embedding_net=graph_embedding_net_config,
        graph_matching_net=graph_matching_net_config,
        model_type=model_type,
        data=dict(
            problem='graph_edit_distance',
            dataset_params=dict(
                # always generate graphs with 20 nodes and p_edge=0.2.
                n_nodes_range=[20, 20],
                p_edge_range=[0.2, 0.2],
                n_changes_positive=0.1,
                n_changes_negative=0.5)),
        training=dict(
            loss='margin',  # other: hamming
            ),
        evaluation=dict(
            batch_size=20),
        seed=8,
    )

def build_datasets(config, communities):
    """Build the training and evaluation datasets."""
    config = copy.deepcopy(config)
    if config['data']['problem'] == 'graph_edit_distance':
        print(f"[数据集构建] 测试社区数: {len(communities)}")
        dataset_params = config['data']['dataset_params']
        dataset_params['dataset_size'] = len(communities)
        validation_set = FixedGraphEditDistanceDataset(**dataset_params, communities=communities)
    else:
        raise ValueError('Unknown problem type: %s' % config['data']['problem'])
    return validation_set

def get_graph(batch):
    if len(batch) != 2:
        # if isinstance(batch, GraphData):
        graph = batch
        node_features = torch.from_numpy(graph.node_features)
        edge_features = torch.from_numpy(graph.edge_features)
        from_idx = torch.from_numpy(graph.from_idx).long()
        to_idx = torch.from_numpy(graph.to_idx).long()
        graph_idx = torch.from_numpy(graph.graph_idx).long()
        return node_features, edge_features, from_idx, to_idx, graph_idx
    else:
        graph, labels = batch
        node_features = torch.from_numpy(graph.node_features)
        edge_features = torch.from_numpy(graph.edge_features)
        from_idx = torch.from_numpy(graph.from_idx).long()
        to_idx = torch.from_numpy(graph.to_idx).long()
        graph_idx = torch.from_numpy(graph.graph_idx).long()
        labels = torch.from_numpy(labels).long()
    return node_features, edge_features, from_idx, to_idx, graph_idx, labels

def reshape_and_split_tensor(tensor, n_splits):
    """Reshape and split a 2D tensor along the last dimension.

    Args:
      tensor: a [num_examples, feature_dim] tensor.  num_examples must be a
        multiple of `n_splits`.
      n_splits: int, number of splits to split the tensor into.

    Returns:
      splits: a list of `n_splits` tensors.  The first split is [tensor[0],
        tensor[n_splits], tensor[n_splits * 2], ...], the second split is
        [tensor[1], tensor[n_splits + 1], tensor[n_splits * 2 + 1], ...], etc..
    """
    feature_dim = tensor.shape[-1]
    tensor = torch.reshape(tensor, [-1, feature_dim * n_splits])
    tensor_split = []
    for i in range(n_splits):
        tensor_split.append(tensor[:, feature_dim * i: feature_dim * (i + 1)])
    return tensor_split


def build_model(config, node_feature_dim, edge_feature_dim):
    """Create model for training and evaluation.

    Args:
      config: a dictionary of configs, like the one created by the
        `get_default_config` function.
      node_feature_dim: int, dimensionality of node features.
      edge_feature_dim: int, dimensionality of edge features.

    Returns:
      tensors: a (potentially nested) name => tensor dict.
      placeholders: a (potentially nested) name => tensor dict.
      AE_model: a GraphEmbeddingNet or GraphMatchingNet instance.

    Raises:
      ValueError: if the specified model or training settings are not supported.
    """
    config['encoder']['node_feature_dim'] = node_feature_dim
    config['encoder']['edge_feature_dim'] = edge_feature_dim

    encoder = GraphEncoder(**config['encoder'])
    aggregator = GraphAggregator(**config['aggregator'])
    if config['model_type'] == 'matching':
        model = GraphMatchingNet(
            encoder, aggregator, **config['graph_matching_net'])
    else:
        raise ValueError('Unknown model type: %s' % config['model_type'])
    return model

def test_model(G, communities, node_embeddings, edge_embeddings, model_path="saved_model.pth"):
    start = time.time()
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    config = get_default_config()
    validation_set = build_datasets(config, communities)
    print("[TIMER] build_datasets:", time.time() - start)

    # first_data_iter = validation_set.pairs(config['evaluation']['batch_size'], G, node_embeddings, edge_embeddings)
    # first_batch_graphs, _ = next(first_data_iter)
    # node_feature_dim = first_batch_graphs.node_features.shape[-1]
    # edge_feature_dim = first_batch_graphs.edge_features.shape[-1]
    node_feature_dim = 30
    edge_feature_dim = 30
    print("[TIMER] first_data_iter:", time.time() - start)

    model = build_model(config, node_feature_dim, edge_feature_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    accumulated_all_metrics = []
    batch_size = config['evaluation']['batch_size']
    total_batches = math.ceil(validation_set._dataset_size * 2 / batch_size)
    print(f"total_batches: {total_batches}")
    for batch_idx, batch in enumerate(validation_set.pairs(config['evaluation']['batch_size'], G, node_embeddings, edge_embeddings)):
        node_features, edge_features, from_idx, to_idx, graph_idx, labels = get_graph(batch)
        labels = labels.to(device)
        edge_index = torch.stack([from_idx, to_idx], dim=0).to(device)
        eval_pairs = model(
            x=node_features.to(device),
            edge_index=edge_index.to(device),
            batch=None,
            graph_idx=graph_idx.to(device),
            edge_features=edge_features.to(device),
            n_graphs=len(torch.unique(graph_idx))
        )
        x, y = reshape_and_split_tensor(eval_pairs, 2)
        similarity = compute_similarity(config, x, y)
        metrics_dict = eval_all_metrics(similarity, labels)
        accumulated_all_metrics.append(metrics_dict)
        metric_str = ', '.join([f"{k}: {v:.4f}" if v is not None else f"{k}: N/A" for k, v in metrics_dict.items()])
        print(f"[Batch {batch_idx + 1}/{total_batches}] {metric_str}")

    all_keys = accumulated_all_metrics[0].keys()
    # 对每个 key 取平均（过滤掉 None）
    avg_metrics = {
        k: np.mean([m[k] for m in accumulated_all_metrics if m[k] is not None])
        for k in all_keys
    }
    print("=== Evaluation Metrics ===")
    print(f"Accuracy:  {avg_metrics['Acc']:.4f}")
    print(f"F1 Score:  {avg_metrics['F1']:.4f}")
    print(f"AUC:       {avg_metrics['AUC']:.4f}")
    print(f"Precision: {avg_metrics['Prec']:.4f}")
    print(f"Recall:    {avg_metrics['Recall']:.4f}")
    print(f"FPR:       {avg_metrics['FPR']:.4f}")
