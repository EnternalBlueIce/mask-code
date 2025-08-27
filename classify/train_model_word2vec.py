# =================训练=========================
import torch
from process.datahandlers import get_handler
from process.embedders import get_embedder_by_name
from process.partition import detect_communities
#from train_utils_word2vec import train_model
from utils_cosine import train_model
from encoder import GIN,MLP_Classifier

class Args:
    lr = 0.001
    epochs = 100
    k = 3
    n_feat = 30
    n_hidden = 64
    n_class = 2
    knn_layer = 1
args=Args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 获取数据集
#data_handler = get_handler("atlas",True)
data_handler = get_handler("theia", True)

# 加载数据
data_handler.load()
# 成整个大图+捕捉特征语料+简化策略这里添加
features, edges, mapp, relations, G  = data_handler.build_graph()
# 大图分割
communities = detect_communities(G)

# 嵌入构造特征向量
embedder_class = get_embedder_by_name("word2vec")
embedder = embedder_class(G, features, mapp)
embedder.train()
node_embeddings = embedder.embed_nodes()
edge_embeddings = embedder.embed_edges()
encoder = GIN(args)
classifier = MLP_Classifier(args)
ground_truth = data_handler.all_labels
train_model(encoder,classifier,communities,node_embeddings,G,args,ground_truth,device=device)
# 模型训练
#train_model(G, communities, node_embeddings, edge_embeddings)
