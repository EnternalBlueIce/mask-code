import os

import igraph as ig
import re
import pandas as pd
import json
from .base import BaseProcessor
from .type_enum import ObjectType
from .common import merge_properties, collect_dot_paths,extract_properties,add_node_properties,get_or_add_node,add_edge_if_new,update_edge_index


class DARPAHandler(BaseProcessor):
    def load(self):
        theia_dir = self.base_path  # 假设为 data_files/theia
        scene_list = [d for d in os.listdir(theia_dir) if
                      d.startswith("theia") and os.path.isdir(os.path.join(theia_dir, d))]

        for scene in scene_list:
            scene_path = os.path.join(theia_dir, scene)

            for category in ["benign", "malicious"]:
                # 文件路径
                txt_filename = f"{scene}_{category}.txt"
                txt_path = os.path.join(theia_dir, txt_filename)
                json_dir = os.path.join(scene_path, category)
                json_files = [os.path.join(json_dir, f) for f in os.listdir(json_dir) if f.endswith(".json")]

                # 数据加载条件控制
                should_load = False
                load_mode = None

                if self.train:
                    if scene == "theia33" and category == "benign":
                        should_load = True
                        load_mode = "train_90"
                    elif scene in ["theia311"] and category == "malicious":
                        should_load = True
                        load_mode = "all"
                else:
                    if scene == "theia33" and category == "benign":
                        should_load = True
                        load_mode = "test_10"
                    elif scene == "theia33" and category == "malicious":
                        should_load = True
                        load_mode = "test"

                if not should_load:
                    continue

                # ✅ 加载训练/测试标签（提前于数据处理）
                if self.train:
                    if scene in ["theia311"] and category == "malicious":
                        label_path = os.path.join(scene_path, "malicious", "label.txt")
                        if os.path.exists(label_path):
                            with open(label_path, "r") as label_file:
                                self.all_labels.extend([line.strip() for line in label_file if line.strip()])
                        else:
                            print(f"[警告] 未找到训练标签文件: {label_path}")
                else:
                    if scene == "theia33" and category == "malicious":
                        label_path = os.path.join(scene_path, "malicious", "label.txt")
                        if os.path.exists(label_path):
                            with open(label_path, "r") as label_file:
                                self.all_labels.extend([line.strip() for line in label_file if line.strip()])
                        else:
                            print(f"[警告] 未找到测试标签文件: {label_path}")

                # 加载日志数据
                if not os.path.exists(txt_path):
                    print(f"[跳过] 未找到日志文件: {txt_path}")
                    continue
                if not json_files:
                    print(f"[跳过] 未找到 JSON 文件: {json_dir}")
                    continue

                with open(txt_path, "r", encoding="utf-8") as f:
                    lines = f.read().splitlines()

                    if load_mode == "train_90":
                        max_count = int(self.max_benign_lines * 0.9)
                        lines = lines[:max_count]
                        print(f"[INFO] 加载 theia33 benign 训练数据: {len(lines)} 行 (90%)")
                    elif load_mode == "test_10":
                        max_count = int(self.max_benign_lines * 0.1)
                        lines = lines[-max_count:]
                        print(f"[INFO] 加载 theia33 benign 测试数据: {len(lines)} 行 (10%)")
                    elif load_mode == "test":
                        #lines = lines[:self.max_test_lines]
                        lines = lines
                        print(f"[INFO] 加载 theia33 malicious 测试数据: {len(lines)} 行 (test)")
                    else:
                        #lines = lines[:self.max_malicious_lines]
                        lines = lines
                        print(f"[INFO] 加载 {scene}_{category}: {len(lines)} 行 (ALL)")

                    data = [line.split('\t') for line in lines]

                df = pd.DataFrame(data, columns=['actorID', 'actor_type', 'objectID', 'object', 'action', 'timestamp'])
                df.dropna(inplace=True)
                df.sort_values(by='timestamp', inplace=True)

                # 构图
                netobj2pro, subject2pro, file2pro = collect_nodes_from_log(json_files)
                df = collect_edges_from_log(df, json_files)

                self.all_dfs.append(df)
                merge_properties(netobj2pro, self.all_netobj2pro)
                merge_properties(subject2pro, self.all_subject2pro)
                merge_properties(file2pro, self.all_file2pro)

        # 拼接数据帧
        if self.all_dfs:
            self.use_df = pd.concat(self.all_dfs, ignore_index=True).drop_duplicates()

    def build_graph(self):
        """成图+捕捉特征语料+简化策略这里添加"""
        G = ig.Graph(directed=True)
        nodes, edges, relations = {}, [], {}

        for _, row in self.use_df.iterrows():
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
            #print(f"actor_id{actor_id}")
            #print(f"object_id{object_id} value{int(object_id in self.all_labels)}")
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


def collect_nodes_from_log(paths):
    netobj2pro = {}
    subject2pro = {}
    file2pro = {}
    for p in paths:
        with open(p,encoding='UTF-8') as f:
            for line in f:
                # --- NetFlowObject ---
                if '{"datum":{"com.bbn.tc.schema.avro.cdm18.NetFlowObject"' in line:
                    try:
                        res = re.findall(
                            'NetFlowObject":{"uuid":"(.*?)"(.*?)"localAddress":"(.*?)","localPort":(.*?),"remoteAddress":"(.*?)","remotePort":(.*?),',
                            line
                        )[0]
                        nodeid = res[0]
                        srcaddr = res[2]
                        srcport = res[3]
                        dstaddr = res[4]
                        dstport = res[5]
                        nodeproperty = f"{srcaddr},{srcport},{dstaddr},{dstport}"
                        netobj2pro[nodeid] = nodeproperty
                    except:
                        pass

                # --- Subject ---
                elif '{"datum":{"com.bbn.tc.schema.avro.cdm18.Subject"' in line:
                    try:
                        res = re.findall(
                            'Subject":{"uuid":"(.*?)"(.*?)"cmdLine":{"string":"(.*?)"}(.*?)"properties":{"map":{"tgid":"(.*?)"',
                            line
                        )[0]
                        nodeid = res[0]
                        cmdLine = res[2]
                        tgid = res[4]
                        try:
                            path_str = re.findall('"path":"(.*?)"', line)[0]
                            path = path_str
                        except:
                            path = "null"
                        nodeProperty = f"{cmdLine},{tgid},{path}"
                        subject2pro[nodeid] = nodeProperty
                    except:
                        pass

                # --- FileObject ---
                elif '{"datum":{"com.bbn.tc.schema.avro.cdm18.FileObject"' in line:
                    try:
                        res = re.findall(
                            'FileObject":{"uuid":"(.*?)"(.*?)"filename":"(.*?)"',
                            line
                        )[0]
                        nodeid = res[0]
                        filepath = res[2]
                        nodeproperty = filepath
                        file2pro[nodeid] = nodeproperty
                    except:
                        pass

    return netobj2pro, subject2pro, file2pro

def collect_edges_from_log(d, paths):
    info = []
    for p in paths:
        with open(p,encoding='UTF-8') as f:
            # TODO
            # for test: 只取每个文件前300条包含"EVENT"的
            #data = [json.loads(x) for i, x in enumerate(f) if "EVENT" in x and i < 10000 ]
            data = [json.loads(x) for i, x in enumerate(f) if "EVENT" in x and i < 50000 ]
        for x in data:
            try:
                action = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['type']
            except:
                action = ''
            try:
                actor = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['subject']['com.bbn.tc.schema.avro.cdm18.UUID']
            except:
                actor = ''
            try:
                obj = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['predicateObject'][
                    'com.bbn.tc.schema.avro.cdm18.UUID']
            except:
                obj = ''
            try:
                timestamp = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['timestampNanos']
            except:
                timestamp = ''
            try:
                cmd = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['properties']['map']['cmdLine']
            except:
                cmd = ''
            try:
                path = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['predicateObjectPath']['string']
            except:
                path = ''
            try:
                path2 = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['predicateObject2Path']['string']
            except:
                path2 = ''
            try:
                obj2 = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['predicateObject2'][
                    'com.bbn.tc.schema.avro.cdm18.UUID']
                info.append({
                    'actorID': actor, 'objectID': obj2, 'action': action, 'timestamp': timestamp,
                    'exec': cmd, 'path': path2
                })
            except:
                pass

            info.append({
                'actorID': actor, 'objectID': obj, 'action': action, 'timestamp': timestamp,
                'exec': cmd, 'path': path
            })

    rdf = pd.DataFrame.from_records(info).astype(str)
    d = d.astype(str)
    rdf.to_csv("rdf_output.csv", index=False, encoding='utf-8')
    d.to_csv("d_output.csv", index=False, encoding='utf-8')

    return d.merge(rdf, how='inner', on=['actorID', 'objectID', 'action', 'timestamp']).drop_duplicates()

