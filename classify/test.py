import torch
from process.datahandlers import get_handler
from process.embedders import get_embedder_by_name
from process.partition import detect_communities
from encoder import GIN,MLP_Classifier
from test_graphs import test_and_evaluate
from types import SimpleNamespace

args = SimpleNamespace(
    n_feat=30,         # Word2Vec 维度
    n_hidden=64,       # GIN 隐层维度
    n_class=2,         # 社区分类类别数（良性/恶性）
    lr=0.001,          # 学习率
    epochs=100,        # 训练轮数
    k=5                # KNN 构图用的邻居数量
)
def load_model(encoder_class, classifier_class, args,
               encoder_path="encoder.pth",
               classifier_path="classifier.pth",
               device=torch.device("cpu")):
    encoder = encoder_class(args).to(device)
    classifier = classifier_class(args).to(device)

    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    classifier.load_state_dict(torch.load(classifier_path, map_location=device))

    encoder.eval()
    classifier.eval()
    print(f"模型已从 '{encoder_path}' 和 '{classifier_path}' 加载完成")
    return encoder, classifier


def save_community_labels(communities, malicious_node_file, save_path="community_labels.txt"):
    # 读取恶意节点集合
    with open(malicious_node_file, 'r') as f:
        malicious_nodes = set(line.strip() for line in f if line.strip())
    with open(save_path, 'w') as f:
        for comm_id in sorted(communities.keys()):
            node_list = communities[comm_id]
            is_malicious = any(node in malicious_nodes for node in node_list)
            label = 1 if is_malicious else 0
            f.write(f"{label}\n")

    print(f"✅ 社区标签已保存到: {save_path}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = SimpleNamespace(
        n_feat=30,  # Word2Vec 维度
        n_hidden=64,  # GIN 隐层维度
        n_class=2,  # 社区分类类别数（良性/恶性）
        lr=0.001,  # 学习率
    )

    # 加载数据
    data_handler = get_handler("theia", False)
    #data_handler = get_handler("atlas", False)
    data_handler.load()
    features, edges, mapp, relations, G = data_handler.build_graph()
    communities = detect_communities(G)
    embedder_class = get_embedder_by_name("word2vec")
    embedder = embedder_class(G, features, mapp)
    embedder.train()
    test_node_embeddings = embedder.embed_nodes()
    ground_truth_malicious_nodes = data_handler.all_labels  # 真实恶意节点名集合

    # 加载训练好的模型
    encoder, classifier = load_model(GIN, MLP_Classifier, args,
                                     encoder_path=r"D:\数据集分析\Flash-IDS-main\process\classify\encoder.pth",
                                     classifier_path=r"D:\数据集分析\Flash-IDS-main\process\classify\classifier.pth",
                                     device=device)

    # 调用测试评估函数
    metrics = test_and_evaluate(encoder, classifier, communities, test_node_embeddings, G,
                                ground_truth_malicious_nodes, prediction_save_path="predicted_labels.txt" ,device=device)

    print("测试评估指标:", metrics)