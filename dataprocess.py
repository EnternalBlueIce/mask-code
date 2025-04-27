from torch.utils.data import Dataset as BaseDataset
from torch_geometric.data.collate import collate
import torch
from utils import *
from torch_geometric.utils import subgraph, degree
from aug import *
from torch_sparse import SparseTensor, matmul
import sys
import traceback
from torch_geometric.utils.mask import index_to_mask
from torch_geometric.utils.num_nodes import maybe_num_nodes
import time


class Dataset_imb(BaseDataset):
    def __init__(self, dataset, all_dataset, args):
        self.args = args
        self.dataset = dataset
        self.all_dataset = all_dataset

    def _get_feed_dict(self, index):
        feed_dict = self.dataset[index]

        return feed_dict

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self._get_feed_dict(index)

    def collate_batch(self, feed_dicts):
        batch_id = torch.tensor([feed_dict.id for feed_dict in feed_dicts])#提取图的id，组成一个张量
        train_idx = torch.arange(batch_id.shape[0])#生成索引
        pad_knn_id = find_knn_id(batch_id, self.args.kernel_idx)#找到KNN邻居节点id
        feed_dicts.extend([self.all_dataset[i] for i in pad_knn_id])#将KNN邻居图也加入到该batch中

        data, slices, _ = collate(
            feed_dicts[0].__class__,
            data_list=feed_dicts,
            increment=True,
            add_batch=True,
        )#将小批次图打包成一张大图

        knn_edge_index, _ = subgraph(
            data.id, self.args.knn_edge_index, relabel_nodes=True)#针对当前id的图提取KNN图

        knn_edge_index, _ = add_remaining_self_loops(knn_edge_index)#添加自环边
        row, col = knn_edge_index
        knn_deg = degree(col, data.id.shape[0])#计算每个节点入度
        deg_inv_sqrt = knn_deg.pow(-0.5)
        edge_weight = deg_inv_sqrt[col] * deg_inv_sqrt[col]#边权重归一化（防止节点特征累加数据爆炸）

        knn_adj_t = torch.sparse.FloatTensor(
            knn_edge_index, edge_weight, (data.id.size(0), data.id.size(0)))#得到该批次内每个节点的KNN连接关系
        
        
        aug_xs, aug_adj_ts = [], []#存放增强后的节点特征，存放增强后的邻接矩阵

        node_map = data.id[data.batch]
        aug_node = self.args.aug_node[node_map]
        aug_edge = self.args.aug_edge[node_map]#得到增强掩码
        row, col = data.adj_t.coo()[:2]
        edge_mask = aug_edge[row] & aug_edge[col]#只有起点和终点都允许增强的边才能被删除

        for i in range(self.args.aug_num):
            edge_index = torch.stack(data.adj_t.coo()[:2])
            edge_index_aug = remove_edge(edge_index, self.args.drop_edge_ratio, edge_mask)#随机删除一部分边
            
            aug_adj_ts.append(SparseTensor(
                row=edge_index_aug[0], col=edge_index_aug[1], value=None, sparse_sizes=(data.x.size(0), data.x.size(0))))
            #重新获得增强后的邻接矩阵
            tmpx = drop_node(data.x, self.args.mask_node_ratio)#随机掩盖一些节点特征
            tmpx[~aug_node] = data.x[~aug_node]#恢复不允许进行数据增强的节点
            aug_xs.append(tmpx)#保存最终增强后的节点特征

        batch = {'data': data,
                 'train_idx': train_idx,
                 'aug_adj_ts': aug_adj_ts,
                 'aug_xs': aug_xs,
                 'knn_adj_t': knn_adj_t}#打包这个batch，包括原始大图，待训练节点索引，邻接矩阵，节点特征集合，KNN邻接矩阵
        return batch
