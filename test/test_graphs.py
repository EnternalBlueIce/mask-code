# =================训练=========================
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from process.datahandlers import get_handler
from process.partition import detect_communities

def load_predictions(file_path):
    predictions = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:  # 确保不是空行
                predictions.append(line)
    return predictions

def evaluate_communities(communities, ground_truths, predictions):
    y_true = []
    y_pred = []
    tp = fp = tn = fn = 0
    for i in range(len(communities)):
        # === 获取该社区的真实标签 ===
        is_true_malicious = any(node in ground_truths for node in communities[i])
        print(is_true_malicious)
        y_true.append(1 if is_true_malicious else 0)
        # === 获取该社区的预测标签 ===
        # 注意：prediction_file 中的顺序应与 communities 中节点顺序一致
        is_pred_malicious = i in predictions
        y_pred.append(1 if is_pred_malicious else 0)

        if is_true_malicious and is_pred_malicious:
            tp += 1
        elif not is_true_malicious and is_pred_malicious:
            fp += 1
        elif is_true_malicious and not is_pred_malicious:
            fn += 1
        elif not is_true_malicious and not is_pred_malicious:
            tn += 1

    # === 计算社区级评估指标 ===
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_pred)
    except ValueError:
        auc = float('nan')
    fpr = fp / (fp + tn + 1e-10)

    print("\n📊 社区级别的评估结果：")
    print(f"✅ Accuracy:  {acc:.4f}")
    print(f"✅ Precision: {prec:.4f}")
    print(f"✅ Recall:    {rec:.4f}")
    print(f"✅ F1 Score:  {f1:.4f}")
    print(f"✅ AUC:       {auc:.4f}")
    print(f"✅ FPR:       {fpr:.4f}")
    print(f"✅ TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
        "fpr": fpr,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn
    }


if __name__ == "__main__":
    #data_handler = get_handler("atlas", False)
    data_handler = get_handler("theia", False)
    data_handler.load()
    features, edges, mapp, relations, G = data_handler.build_graph()
    communities = detect_communities(G)
    prediction_path = r"D:\数据集分析\Flash-IDS-main\process\classify\predicted_labels.txt"
    predictions = load_predictions(prediction_path)
    result = evaluate_communities(communities, data_handler.all_labels, predictions)


