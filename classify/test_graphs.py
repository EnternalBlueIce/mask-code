import torch
from sklearn.metrics import accuracy_score, f1_score
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

    # 构建社区批次数据
    batch_data, _ = build_community_batch(communities, node_embeddings, G, device)

    # 编码社区图，得到社区表示
    graph_embeds = encoder(batch_data.x, batch_data.edge_index, batch_data.batch)
    out = classifier(graph_embeds)  # (num_communities, num_classes)
    probs = torch.softmax(out, dim=1)
    malignant_probs = probs[:, 1]
    print(probs)
    # 2. 设定阈值
    preds = (malignant_probs >= 0.5).long()
    # 得到预测类别（0 或 1）
    #preds = out.argmax(dim=1).cpu().numpy().tolist()
    # 保存预测标签到文件
    with open(prediction_save_path, 'w') as f:
        for label in preds:
            f.write(f"{label}\n")

    print(f"✅ 预测结果已保存到：{prediction_save_path}")

    # 构造预测的恶意社区索引列表
    predicted_malicious_communities = [i for i, label in enumerate(preds) if label == 1]
    print("预测为恶意的社区索引：", predicted_malicious_communities)

    # 使用已有函数计算社区级评估指标
    metrics = evaluate_communities(communities, ground_truth_malicious_nodes, predicted_malicious_communities)

    return metrics

