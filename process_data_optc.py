import os
import re
import json
import warnings
import torch
from datetime import datetime, time


def extract_uuid(line): pass
def extract_subject_type(line): pass
def extract_edge_info(line): pass
def show(file_path): print(f"Processing {file_path}")    #辅助函数，用于打印处理的文件路径
#
def ensure_dir(path: str):
    if not os.path.isdir(path): os.makedirs(path, exist_ok=True)

TARGET_DATE = datetime(2019, 9, 23).date()
LOWER_T = time(11, 23, 29)
UPPER_T = time(15, 30, 0)
#负责安全地将各种格式的时间戳字符串转换为Python的 datetime 对象
def _parse_ts(ts: str):
    if not ts: return None
    s = ts.strip()
    if s.endswith('Z'): s = s[:-1] + '+00:00'
    try: return datetime.fromisoformat(s)
    except Exception: return None
#使用 _parse_ts 的结果来判断一个时间戳是否落在 TARGET_DATE 和 LOWER_T/UPPER_T 定义的范围内。
def _in_time_window(ts: str) -> bool:
    dt = _parse_ts(ts)
    if dt is None: return False
    local_date = dt.date()
    local_time = dt.timetz().replace(tzinfo=None)
    if local_date != TARGET_DATE: return False
    return LOWER_T <= local_time <= UPPER_T

def filter_json_by_hostname_and_time(file_path, keyword="SysClient0201"):
    out_dir = os.path.join(os.path.dirname(file_path), "filtered")
    ensure_dir(out_dir) #防止输出文件夹不存在
    base = os.path.basename(file_path) #从完整路径中提取出文件名部分
    name, _ext = os.path.splitext(base) #将文件名分割成两部分：文件名主体和扩展名。
    # 文件名仍然可以包含时间窗口信息以作标识
    time_window_str = f"{TARGET_DATE.strftime('%Y%m%d')}_{LOWER_T.strftime('%H%M%S')}-{UPPER_T.strftime('%H%M%S')}"  #用来表示由全局变量定义的时间窗口 (修正：增加了秒)
    out_json = os.path.join(out_dir, f"{name}_{keyword}_{time_window_str}.json") #构建了最终输出的JSON文件的完整路径

    kept_objs = []
    total = 0
    hostname_pat = re.compile(r'"hostname"\s*:\s*"([^"]*)"')
    ts_pat = re.compile(r'"timestamp"\s*:\s*"([^"]*)"')
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as fr:
        for line in fr:
            if not line.strip(): continue
            total += 1
            try:
                obj = json.loads(line.strip().rstrip(','))
                if ("hostname" in obj and keyword in str(obj["hostname"])
                        and _in_time_window(obj.get("timestamp"))):
                    kept_objs.append(obj)
                continue
            except Exception:
                #异常处理
                host_m = hostname_pat.search(line)
                if not host_m or keyword not in host_m.group(1): continue #如果正则表达式没有找到主机名 (not host_m)，或者找到的主机名中不包含 keyword，那么 continue 语句会跳过这一行，继续处理下一行
                ts_m = ts_pat.search(line)
                ts_val = ts_m.group(1) if ts_m else None
                if _in_time_window(ts_val):
                    try:
                        obj = json.loads(line.strip().rstrip(','))
                        kept_objs.append(obj)
                    except Exception: pass
    with open(out_json, "w", encoding="utf-8") as fw: #写入文件
        json.dump(kept_objs, fw, ensure_ascii=False, separators=(',', ':')) #它接收整个 kept_objs 列表（里面全是Python字典），并将其作为一个格式规范的JSON数组写入到 fw 文件中
    print(f"[Filter] {file_path} -> {out_json} | kept {len(kept_objs)}/{total}")
    return out_json
#处理点
def process_data(file_path):
    id_nodetype_map = {} #初始化一个空的字典，名为 `id_nodetype_map`。它将用来存储键值对，格式为 `{'节点ID': '节点类型'}`
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f: #read
            data = json.load(f) #json.load (注意，不是 loads) 会尝试一次性读取并解析整个文件 f。它期望这个文件是一个完整的、合法的JSON结构，比如一个由 [ 开始、由 ] 结束的JSON数组。如果成功，所有数据会被加载到 data 变量中，它会是一个包含多个字典的Python列表
        if isinstance(data, list): #这是一个验证步骤，检查从文件中加载的 data 是否确实是一个列表（即JSON数组）
            for obj in data: #如果 data 是一个列表，就开始遍历这个列表中的每一个元素。这里的每个 obj 都应该是一个代表单条日志记录的Python字典
                if not isinstance(obj, dict): continue #再次验证，确保列表中的元素 obj 是一个字典。如果不是（例如，列表中混入了 null 或其他类型的值），就用 continue 跳过这个元素
                actor = obj.get("actorID") #使用 .get() 方法安全地获取 "actorID" 的值。如果这个键不存在，.get() 会返回 None 而不是报错。
                if actor: id_nodetype_map[actor] = "Subject"  #如果成功获取到了 actor 的ID（即值不是None），就将其作为键添加到 id_nodetype_map 中，并将其类型硬编码为 "Subject"。
                object_id = obj.get("objectID")
                object_type = obj.get("object")
                if object_id and object_type: id_nodetype_map[object_id] = object_type  # 只有当客体的ID和类型都成功获取到时，才将其添加到节点地图中
            return id_nodetype_map
    except Exception: pass #如果try出错执行这个代码，意味着什么都不做
    #如果程序执行到这里，说明输入文件不是一个标准的JSON数组，需要用更“原始”的方法来处理
    actor_pattern = re.compile(r'"actorID":"([^"]*)"')
    object_id_pattern = re.compile(r'"objectID":"([^"]*)"')
    object_type_pattern = re.compile(r'"object":"([^"]*)"')
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            actor_match = actor_pattern.search(line) #在当前行 line 中搜索匹配 actor_pattern 的部分
            if actor_match: id_nodetype_map[actor_match.group(1)] = 'Subject' #如果找到了匹配项
            object_id_match = object_id_pattern.search(line)  #同上
            object_type_match = object_type_pattern.search(line)
            if object_id_match and object_type_match:
                id_nodetype_map[object_id_match.group(1)] = object_type_match.group(1) #加上字典
    return id_nodetype_map
#处理边
def process_edges_and_count(file_path, id_nodetype_map, output_path):
    #用于统计总共成功提取并写入了多少条“边”
    edge_count = 0
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f: data = json.load(f)
        if isinstance(data, list):
            with open(output_path, 'a', encoding='utf-8') as fw:
                for obj in data:
                    if not isinstance(obj, dict): continue
                    src_id, dst_id, edge_type, ts = obj.get("actorID"), obj.get("objectID"), obj.get("action"), obj.get("timestamp") #在一行内从 `obj` 字典中提取出构建一条“边”所需的全部四个要素
                    if src_id and dst_id and edge_type and ts  and src_id in id_nodetype_map and dst_id in id_nodetype_map:#src_id and dst_id and edge_type and ts: 确保源、目标、类型、时间戳这四个基本要素都必须存在,src_id in id_nodetype_map and dst_id in id_nodetype_map: 确保这条边的源节点ID和目标节点ID都存在于我们传入的“节点地图” id_nodetype_map 中
                        src_type, dst_type = id_nodetype_map[src_id], id_nodetype_map[dst_id] #如果 if 条件全部满足，就从 id_nodetype_map 中查询出源节点和目标节点的类型
                        fw.write(f"{src_id}\t{src_type}\t{dst_id}\t{dst_type}\t{edge_type}\t{ts}\n") #使用 f-string 创建一个格式化的字符串，字段之间用制表符 \t 分隔，并在末尾加上换行符 \n。然后，使用 fw.write() 将这行文本写入到输出文件中
                        edge_count += 1
            return edge_count
    except Exception: pass
    actor_pattern = re.compile(r'"actorID":"([^"]*)"')
    object_pattern = re.compile(r'"objectID":"([^"]*)"')
    action_pattern = re.compile(r'"action":"([^"]*)"')
    timestamp_pattern = re.compile(r'"timestamp":"([^"]*)"')
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f, open(output_path, 'a', encoding='utf-8') as fw:
        for line in f:
            actor, obj, action, timestamp = actor_pattern.search(line), object_pattern.search(line), action_pattern.search(line), timestamp_pattern.search(line) #在一行内，使用四个正则表达式分别在当前行 line 中进行搜索，并将匹配结果对象（如果匹配成功）或 None（如果匹配失败）赋值给相应的变量
            if actor and obj and action and timestamp:
                src_id, dst_id, edge_type, ts = actor.group(1), obj.group(1), action.group(1), timestamp.group(1) #如果都匹配成功，就从每个匹配对象中提取出第一个捕获组的内容（即ID、类型和时间戳的值）
                if src_id in id_nodetype_map and dst_id in id_nodetype_map: #是否在全局地图中
                    src_type, dst_type = id_nodetype_map[src_id], id_nodetype_map[dst_id]
                    fw.write(f"{src_id}\t{src_type}\t{dst_id}\t{dst_type}\t{edge_type}\t{ts}\n")
                    edge_count += 1
    return edge_count

def collect_json_paths(base_dir):
    result = {'optc_data_scene': {'all_data': []}}
    if not os.path.isdir(base_dir):
        print(f"Error: Directory '{base_dir}' not found.")
        return {}
    for file in os.listdir(base_dir):
        if file.endswith((".json", ".txt", ".log")) and not file.startswith("._"):
            result['optc_data_scene']['all_data'].append(os.path.join(base_dir, file))
    return result


def run_data_processing():
    base_path = r"D:\数据集分析\Flash-IDS-main\optc"
    json_map = collect_json_paths(base_path)

    statistics = {"total_nodes": 0, "total_edges": 0}

    print("\n========= Stage 1: Filtering All Files to Generate Subset JSONs =========")
    all_filtered_paths = []
    for scene, data in json_map.items():
        for category in data.keys():
            for original_path in data.get(category, []):
                filtered_path = filter_json_by_hostname_and_time(original_path, keyword="SysClient0201")
                all_filtered_paths.append(filtered_path)
    print("\nStage 1 Complete. All filtered JSON files have been generated.")

    print("\n========= Stage 2: Building Individual Graphs from Each Filtered File =========")

    print("--- Building a combined node map from ALL filtered files...")
    combined_nodetype_map = {}
    for filtered_path in all_filtered_paths:
        id_map_from_file = process_data(filtered_path)
        combined_nodetype_map.update(id_map_from_file)
    statistics["total_nodes"] = len(combined_nodetype_map)
    print(f"Combined node map built. Total unique nodes in filtered data: {statistics['total_nodes']}")

    for filtered_path in all_filtered_paths:

        dir_name = os.path.dirname(filtered_path)
        base_name = os.path.basename(filtered_path)
        name, _ext = os.path.splitext(base_name)
        output_txt_path = os.path.join(dir_name, f"{name}.txt")

        print(f"\n--- Processing: {filtered_path} -> {output_txt_path} ---")

        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write("Source_ID\tSource_Type\tDestination_ID\tDestination_Type\tEdge_Type\tTimestamp\n")

        edge_cnt = process_edges_and_count(filtered_path, combined_nodetype_map, output_txt_path)
        statistics["total_edges"] += edge_cnt
        print(f"Edges extracted: {edge_cnt}")

    print("\n========= Processing Complete =========")
    print(f"Total unique nodes across all filtered files: {statistics['total_nodes']}")
    print(f"Total edges extracted across all filtered files: {statistics['total_edges']}")
    print("Individual TXT graph files have been generated next to their corresponding filtered JSON files.")


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_data_processing()