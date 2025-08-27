import torch
from process.test.test_graphs import evaluate_communities
from utils_cosine import build_community_batch

@torch.no_grad()
def test_and_evaluate(encoder, classifier, communities, node_embeddings, G,
                      ground_truth_malicious_nodes,
                      prediction_save_path="predicted_labels.txt",
                      device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    encoder.to(device)
    classifier.to(device)
    encoder.eval()
    classifier.eval()

    # 构建社区批次数据：获取社区中每个节点的 node_id 列表
    batch_data, community_node_ids = build_community_batch(communities, node_embeddings, G, device)

    # 不再使用 encoder 得到图嵌入，直接使用 node_embeddings
    # 注意：假设 node_embeddings 是一个 (num_nodes, emb_dim) 的 Tensor
    # 从中挑选出 batch 中的节点嵌入
    node_embeds = encoder(batch_data.x, batch_data.edge_index,batch_data.batch)  # (num_nodes_in_batch, emb_dim)

    # 使用节点级分类器对每个节点预测
    node_outputs = classifier(node_embeds)  # (num_nodes_in_batch, 2)
    node_preds = node_outputs.argmax(dim=1).cpu().numpy()  # (num_nodes_in_batch,)

    # 构造每个社区的预测结果：如果有一个节点为异常，整个社区为异常
    predicted_malicious_communities = []
    community_labels = []

    start_idx = 0
    for i, node_ids in enumerate(community_node_ids):
        preds = node_preds[node_ids]

        if any(p == 1 for p in preds):
            community_labels.append(1)
            predicted_malicious_communities.append(i)
        else:
            community_labels.append(0)

    # 保存社区标签到文件
    with open(prediction_save_path, 'w') as f:
        for label in community_labels:
            f.write(f"{label}\n")

    print(f"✅ 社区预测结果已保存到：{prediction_save_path}")
    print("预测为恶意的社区索引：", predicted_malicious_communities)

    # 调用已有评估函数，计算指标
    metrics = evaluate_communities(communities, ground_truth_malicious_nodes, predicted_malicious_communities)

    return metrics