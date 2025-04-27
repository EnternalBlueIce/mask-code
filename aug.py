import numpy as np
import torch
from torch_geometric.utils.dropout import dropout_edge
import time
import copy
import random

def dropout_edge(edge_index, p=0.5, edge=None):#默认丢弃比例50%
    if p == 0.0:
        edge_mask = edge_index.new_ones(edge_index.size(1), dtype=torch.bool)
        return edge_index, edge_mask#如果丢弃比例为0，就让所有的edge_mask等于1，表示不丢弃任何边

    row, col = edge_index
    # deal edge
    edge_mask = torch.rand(row.size(0), device=edge_index.device) < p
    if edge is not None:
        edge_mask = torch.logical_and(edge_mask, edge)
    
    edge_index = edge_index[:, ~edge_mask]

    return edge_index, edge_mask


def remove_edge(edge_index, drop_ratio, edge=None):
    edge_index, _ = dropout_edge(edge_index, p = drop_ratio, edge=edge)#按比例丢弃边
    return edge_index#返回丢弃后的索引张量


def drop_node(x, drop_ratio):
    node_num, _ = x.size()#提取节点数量
    drop_num = int(node_num * drop_ratio)#计算被掩盖节点

    idx_mask = np.random.choice(node_num, drop_num, replace = False).tolist()#随机选择drop_num个节点

    x[idx_mask] = 0 #讲这些其被选择的节点掩码值设置为0

    return x#返回全部节点
