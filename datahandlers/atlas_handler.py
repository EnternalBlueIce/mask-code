import os.path

from .common import merge_properties, collect_dot_paths, extract_properties, add_node_properties, get_or_add_node, \
    add_edge_if_new, update_edge_index, collect_atlas_label_paths
from .base import BaseProcessor
import re
import pandas as pd
import igraph as ig

from .type_enum import ObjectType


class ATLASHandler(BaseProcessor):
    def load(self):
        print("处理 ATLAS 数据集...")
        graph_files = collect_dot_paths(self.base_path)
        label_map = collect_atlas_label_paths(self.base_path)

        processed_data = []
        domain_name_set = {}
        ip_set = {}
        connection_set = {}
        session_set = {}
        web_object_set = {}
        # 处理每个 .dot 文件

        for dot_file in graph_files:
            filename = os.path.basename(dot_file)
            # TODO test
            # if "M3-CVE-2015-3105_windows_py_h2" not in dot_file :
            #     continue
            self.total_loaded_bytes += os.path.getsize(dot_file)
            print(f"正在处理文件: {dot_file}")
            file_with_ext = os.path.basename(dot_file)
            dot_name = os.path.splitext(file_with_ext)[0]
            if dot_name.startswith("graph_"):
                dot_name = dot_name[len("graph_"):]
            if self.train == False:
                #print(label_map)
                label_file = open(label_map[dot_name])

                print(f"正在处理: 场景={dot_name}, label={label_file}")
                self.all_labels.extend([
                    line.strip() for line in label_file.read().splitlines() if line.strip()
                ])

            # 读取节点数据
            netobj2pro, subject2pro, file2pro, domain_name_set, ip_set, connection_set, session_set, web_object_set = collect_nodes_from_log(
                dot_file)

            # 打印节点数据
            #print("netobj2pro:", netobj2pro)
            #print("subject2pro:", subject2pro)
            #print("file2pro:", file2pro)
            print("正在处理边: ")
            # 调用 `collect_edges_from_log` 收集边
            df = collect_edges_from_log(dot_file, domain_name_set, ip_set, connection_set, session_set, web_object_set,
                                        subject2pro, file2pro)  # 将 dot 文件传入收集边的函数
            df.to_csv("output_{}.csv".format(filename), index=False)
            print("edges:", df)

            # 只取良性前90%训练
            num_rows = int(len(df) * 0.9)
            df = df.iloc[:num_rows]  # 正确缩进
            self.all_dfs.append(df)
            merge_properties(netobj2pro, self.all_netobj2pro)
            merge_properties(subject2pro, self.all_subject2pro)
            merge_properties(file2pro, self.all_file2pro)

        # 训练用的数据集
        # 将多个 DataFrame 合并成一个大的 DataFrame且去重
        self.all_dfs = pd.concat(self.all_dfs, ignore_index=True)
        self.all_dfs = self.all_dfs.drop_duplicates()

        # TODO test
        self.all_dfs = self.all_dfs.drop_duplicates().iloc[:500]
        # 将处理好后的数据集保存到 ATLAS.txt
        self.all_dfs.to_csv("ATLAS.txt", sep='\t', index=False)

    # 成图+捕捉特征语料+简化策略这里添加
    def build_graph(self):
        G = ig.Graph(directed=True)
        nodes, edges, relations = {}, [], {}

        for _, row in self.all_dfs.iterrows():
            action = row["action"]

            actor_id = row["actorID"]
            properties = extract_properties(actor_id, row, row["action"], self.all_netobj2pro, self.all_subject2pro, self.all_file2pro)
            add_node_properties(nodes, actor_id, properties)

            object_id = row["objectID"]
            properties1 = extract_properties(object_id, row, row["action"], self.all_netobj2pro, self.all_subject2pro,
                                             self.all_file2pro)
            add_node_properties(nodes, object_id, properties1)

            edge = (actor_id, object_id)
            edges.append(edge)
            relations[edge] = action

            ## 构建图
            # 点不重复添加
            actor_idx = get_or_add_node(G, actor_id, ObjectType[row['actor_type']].value, properties)
            object_idx = get_or_add_node(G, object_id, ObjectType[row['object']].value, properties)
            # 标注label
            print(f"actor_id{actor_id}")
            print(f"object_id{object_id} value{int(object_id in self.all_labels)}")
            G.vs[actor_idx]["label"] = int(actor_id in self.all_labels)
            G.vs[object_idx]["label"] = int(object_id in self.all_labels)
            # 边也不重复添加
            add_edge_if_new(G, actor_idx, object_idx, action)

        features, edge_index, index_map, relations_index = [], [[], []], {}, {}
        for node_id, props in nodes.items():
            features.append(props)
            index_map[node_id] = len(features) - 1

        update_edge_index(edges, edge_index, index_map, relations, relations_index)

        return features, edge_index, list(index_map.keys()), relations_index, G


def collect_nodes_from_log(paths):  # dot文件的路径
    # 创建字典
    netobj2pro = {}
    subject2pro = {}
    file2pro = {}
    domain_name_set = {}
    ip_set = {}
    connection_set = {}
    session_set = {}
    web_object_set = {}
    nodes = []

    # 读取整个文件
    with open(paths, 'r', encoding='utf-8') as f:
        content = f.read()

    # 按分号分隔，处理每个段落
    statements = content.split(';')

    # 正则表达式匹配节点定义
    node_pattern = re.compile(r'^\s*"?(.+?)"?\s*\[.*?type="?([^",\]]+)"?', re.IGNORECASE)

    for stmt in statements:
        if 'capacity=' in stmt:
            continue  # 跳过包含 capacity 字段的段落
        match = node_pattern.search(stmt)
        if match:
            node_name = match.group(1)
            node_typen = match.group(2)
            nodes.append((node_name, node_typen))
    for node_name, node_typen in nodes:  # 遍历所有的节点
        node_id = node_name  # 节点id赋值
        node_type = node_typen  # 赋值type属性
        # -- 网络流节点 --
        if node_type == 'domain_name':
            nodeproperty = node_id
            netobj2pro[node_id] = nodeproperty
            domain_name_set[node_id] = nodeproperty
        if node_type == 'IP_Address':
            nodeproperty = node_id
            netobj2pro[node_id] = nodeproperty
            ip_set[node_id] = nodeproperty
        if node_type == 'connection':
            nodeproperty = node_id
            netobj2pro[node_id] = nodeproperty
            connection_set[node_id] = nodeproperty
        if node_type == 'session':
            nodeproperty = node_id
            netobj2pro[node_id] = nodeproperty
            session_set[node_id] = nodeproperty
        if node_type == 'web_object':
            nodeproperty = node_id
            netobj2pro[node_id] = nodeproperty
            web_object_set[node_id] = nodeproperty
        # -- 进程节点 --
        elif node_type == 'process':
            nodeproperty = node_id
            subject2pro[node_id] = nodeproperty
        # -- 文件节点 --
        elif node_type == 'file':
            nodeproperty = node_id
            file2pro[node_id] = nodeproperty

    return netobj2pro, subject2pro, file2pro, domain_name_set, ip_set, connection_set, session_set, web_object_set


def collect_edges_from_log(paths, domain_name_set, ip_set, connection_set, session_set, web_object_set, subject2pro,
                           file2pro) -> pd.DataFrame:
    """
    从 DOT-like 日志文件中提取含 capacity 的边，并识别 source/target 属于哪个节点集合。
    返回一个包含 source、target、type、timestamp、source_type、target_type 的 DataFrame。
    """
    # 预定义的节点集合

    edges = []

    with open(paths, "r", encoding="utf-8") as f:
        content = f.read()

    statements = content.split(";")

    edge_pattern = re.compile(
        r'"?([^"]+)"?\s*->\s*"?(.*?)"?\s*\['
        r'.*?capacity=.*?'
        r'type="?([^",\]]+)"?.*?'
        r'timestamp=(\d+)',
        re.IGNORECASE | re.DOTALL
    )

    for stmt in statements:
        if "capacity=" not in stmt:
            continue
        m = edge_pattern.search(stmt)
        if m:
            source, target, edge_type, ts = (x.strip() for x in m.groups())

            # 判断 source/target 所属集合
            if source in domain_name_set:
                source_type = "NETFLOW_OBJECT"
            elif source in ip_set:
                source_type = "NETFLOW_OBJECT"
            elif source in connection_set:
                source_type = "NETFLOW_OBJECT"
            elif source in session_set:
                source_type = "NETFLOW_OBJECT"
            elif source in web_object_set:
                source_type = "NetFlowObject"
            elif source in subject2pro:
                source_type = "SUBJECT_PROCESS"
            elif source in file2pro:
                source_type = "FILE_OBJECT_BLOCK"
            else:
                source_type = "PRINCIPAL_LOCAL"

            if target in domain_name_set:
                target_type = "NETFLOW_OBJECT"
            elif target in ip_set:
                target_type = "NETFLOW_OBJECT"
            elif target in connection_set:
                target_type = "NETFLOW_OBJECT"
            elif target in session_set:
                target_type = "NETFLOW_OBJECT"
            elif target in web_object_set:
                target_type = "NetFlowObject"
            elif target in subject2pro:
                target_type = "SUBJECT_PROCESS"
            elif target in file2pro:
                target_type = "FILE_OBJECT_BLOCK"
            else:
                target_type = "PRINCIPAL_LOCAL"

            edges.append((source, source_type, target, target_type, edge_type, int(ts)))

    return pd.DataFrame(edges, columns=["actorID", "actor_type", "objectID", "object", "action", "timestamp"])
