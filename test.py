# =================训练=========================
import torch
from datahandlers import get_handler
from embedders import get_embedder_by_name
from process.match.test_model import test_model
from process.partition import detect_communities

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 获取测试数据集
data_handler = get_handler("atlas", False)
#data_handler = get_handler("theia", False)
data_handler.load()

# 成整个大图+捕捉特征语料+简化策略这里添加
features, edges, mapp, relations, G = data_handler.build_graph()

# 大图分割
communities = detect_communities(G)

# 嵌入构造特征向量
embedder_class = get_embedder_by_name("word2vec")
# embedder_class = get_embedder_by_name("transe")
embedder = embedder_class(G, features, mapp)
embedder.train()
node_embeddings = embedder.embed_nodes()
edge_embeddings = embedder.embed_edges()

# 模型测试
test_model(G, communities, node_embeddings, edge_embeddings)

