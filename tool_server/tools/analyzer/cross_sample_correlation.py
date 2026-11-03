"""
cross_sample_correlation — 跨样本攻击模板发现
============================================
检测批量投毒中同一攻击模板被复用于多个样本的模式。

算法：
1. 对每个样本提取字符级 N-gram 指纹
2. 计算样本间 Jaccard 相似度
3. 构建相似度图 → 找连通分量（模板簇）
4. 检测簇内标签倾斜 + 提取共享模板签名

调用方式：
    scan_dataset(dataset, options) → dict
"""

import re
from collections import Counter


# ═════════════════════════════════════════════════════════
# 结构指纹提取
# ═════════════════════════════════════════════════════════

def _extract_fingerprint(text: str, ngram_size: int = 4) -> set[str]:
    """
    提取文本的字符级 N-gram 指纹集合。
    统一转小写，跳过纯数字/纯符号。
    """
    text_lower = text.lower().strip()
    if len(text_lower) < ngram_size:
        return {text_lower}  # 短文本直接用全文

    ngrams = set()
    for i in range(len(text_lower) - ngram_size + 1):
        chunk = text_lower[i:i + ngram_size]
        # 跳过纯数字/空白
        if chunk.strip() and not chunk.isdigit():
            ngrams.add(chunk)

    return ngrams


def _extract_structural_features(text: str) -> dict:
    """提取结构特征用于辅助判断。"""
    return {
        "char_count": len(text),
        "word_count": len(text.split()),
        "sentence_count": len(re.split(r'[。！？.!?]', text)),
        "has_english": bool(re.search(r'[a-zA-Z]{3,}', text)),
        "has_chinese": bool(re.search(r'[一-鿿]', text)),
    }


# ═════════════════════════════════════════════════════════
# Jaccard 相似度
# ═════════════════════════════════════════════════════════

def _jaccard_similarity(set_a: set, set_b: set) -> float:
    """计算两个集合的 Jaccard 相似度。"""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


# ═════════════════════════════════════════════════════════
# 图聚类（连通分量）
# ═════════════════════════════════════════════════════════

def _find_clusters(similarity_matrix: list[list[float]], threshold: float, min_size: int) -> list[list[int]]:
    """
    基于相似度矩阵的简单图聚类（连通分量 + 过滤）。
    返回簇列表，每个簇是一组索引。
    """
    n = len(similarity_matrix)

    # 构建邻接表
    adj = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if similarity_matrix[i][j] >= threshold:
                adj[i].add(j)
                adj[j].add(i)

    # 找连通分量（BFS）
    visited = set()
    clusters = []

    for start in range(n):
        if start in visited:
            continue

        # BFS
        component = []
        queue = [start]
        visited.add(start)

        while queue:
            node = queue.pop(0)
            component.append(node)
            for neighbor in adj[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        if len(component) >= min_size:
            clusters.append(sorted(component))

    return clusters


# ═════════════════════════════════════════════════════════
# 模板签名提取
# ═════════════════════════════════════════════════════════

def _extract_template_signature(fingerprints: list[set], indices: list[int], top_k: int = 5) -> list[str]:
    """
    从簇内样本中提取共享的模板 N-gram 签名。
    找在簇内高频、但在簇外低频的 N-gram。
    """
    if len(indices) < 2:
        return []

    # 簇内 N-gram 频率
    cluster_ngrams = Counter()
    for idx in indices:
        cluster_ngrams.update(fingerprints[idx])

    # 簇外 N-gram 频率
    outside_ngrams = Counter()
    for idx in range(len(fingerprints)):
        if idx not in indices:
            outside_ngrams.update(fingerprints[idx])

    # 找"簇内高频 + 簇外低频"的 N-gram
    cluster_size = len(indices)
    signature_scores = {}
    for ngram, cluster_freq in cluster_ngrams.items():
        if cluster_freq < max(2, cluster_size * 0.5):
            continue  # 至少一半的簇内样本共享
        outside_freq = outside_ngrams.get(ngram, 0)
        # 簇内占比远高于簇外
        cluster_ratio = cluster_freq / cluster_size
        outside_ratio = outside_freq / max(len(fingerprints) - cluster_size, 1)
        if cluster_ratio > 0.5 and cluster_ratio > outside_ratio * 2:
            signature_scores[ngram] = cluster_ratio

    # 返回 top_k 个最有代表性的
    sorted_sigs = sorted(signature_scores.items(), key=lambda x: -x[1])
    return [sig for sig, score in sorted_sigs[:top_k]]


# ═════════════════════════════════════════════════════════
# 公开 API
# ═════════════════════════════════════════════════════════

def scan_dataset(dataset: list, options: dict = None) -> dict:
    """
    跨样本攻击模板检测。

    参数:
        dataset: [{"text": "...", "label": "..."}, ...] 或 ["text", ...]
        options: {
            "similarity_threshold": 0.6,
            "min_cluster_size": 3,
            "ngram_size": 4,
        }

    返回:
        {
            "total_samples": int,
            "template_clusters": [...],
            "risk_score": 0-100,
            "summary": str,
        }
    """
    if options is None:
        options = {}

    threshold = options.get("similarity_threshold", 0.6)
    min_cluster_size = options.get("min_cluster_size", 3)
    ngram_size = options.get("ngram_size", 4)

    # 统一格式
    texts = []
    labels = []
    for item in dataset:
        if isinstance(item, str):
            texts.append(item)
            labels.append("unknown")
        elif isinstance(item, dict):
            texts.append(item.get("text", item.get("content", str(item))))
            labels.append(item.get("label", "unknown"))
        else:
            texts.append(str(item))
            labels.append("unknown")

    n = len(dataset)

    if n < min_cluster_size:
        return {
            "total_samples": n,
            "template_clusters": [],
            "risk_score": 0.0,
            "summary": f"样本数不足（{n} < {min_cluster_size}），跳过跨样本分析",
        }

    # ── 1. 提取指纹 ──
    fingerprints = [_extract_fingerprint(t, ngram_size) for t in texts]
    structures = [_extract_structural_features(t) for t in texts]

    # ── 2. 计算相似度矩阵 ──
    # 只计算上三角，O(n²) 对于 n<2000 可接受
    sim_matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            sim = _jaccard_similarity(fingerprints[i], fingerprints[j])
            sim_matrix[i][j] = sim
            sim_matrix[j][i] = sim

    # ── 3. 图聚类 ──
    clusters = _find_clusters(sim_matrix, threshold, min_cluster_size)

    # ── 4. 分析每个簇 ──
    # 全局标签分布
    global_label_dist = Counter(labels)
    global_n = n

    template_clusters = []
    for cluster_indices in clusters:
        # 标签偏斜度
        cluster_labels = [labels[i] for i in cluster_indices]
        cluster_label_dist = Counter(cluster_labels)
        max_label, max_count = cluster_label_dist.most_common(1)[0]
        cluster_purity = max_count / len(cluster_indices)

        # 该标签在全局的比例
        global_ratio = global_label_dist.get(max_label, 0) / max(global_n, 1)

        # 如果簇内纯度高且偏离全局 → 模板化攻击
        label_skew = cluster_purity - global_ratio
        is_attack = label_skew > 0.3 and cluster_purity > 0.7

        # 提取共享模板
        template_ngrams = _extract_template_signature(fingerprints, cluster_indices)

        # 置信度
        if is_attack:
            confidence = min(0.95, 0.6 + label_skew * 0.5 + len(template_ngrams) * 0.05)
        else:
            confidence = max(0.1, 0.3 + label_skew * 0.3)

        # 重建共享模板文本
        shared_template = ""
        if template_ngrams:
            # 从第一个样本中截取包含最多模板 N-gram 的片段
            first_text = texts[cluster_indices[0]].lower()
            best_snippet = first_text[:200]
            shared_template = best_snippet

        template_clusters.append({
            "cluster_id": len(template_clusters) + 1,
            "size": len(cluster_indices),
            "indices": cluster_indices,
            "shared_template": shared_template[:200],
            "template_ngrams": template_ngrams[:10],
            "label_skew": round(label_skew, 3),
            "is_attack_template": is_attack,
            "confidence": round(confidence, 2),
            "representative_samples": [texts[i][:80] for i in cluster_indices[:3]],
        })

    # 按是否为攻击模板 + 置信度排序
    template_clusters.sort(key=lambda x: (-x["is_attack_template"], -x["confidence"]))

    # ── 5. 风险评分 ──
    attack_clusters = [c for c in template_clusters if c["is_attack_template"]]
    if attack_clusters:
        affected = sum(c["size"] for c in attack_clusters)
        max_conf = max(c["confidence"] for c in attack_clusters)
        risk_score = min(max_conf * 80 + min(affected / max(n, 1) * 100, 40), 100.0)
    else:
        risk_score = 0.0

    # ── 6. 汇总 ──
    attack_n = len(attack_clusters)
    total_affected = sum(c["size"] for c in template_clusters)

    if attack_n > 0:
        summary = (
            f"发现{attack_n}个攻击模板簇，"
            f"共{total_affected}个样本被聚类，"
            f"受影响样本{sum(c['size'] for c in attack_clusters)}个"
        )
    elif template_clusters:
        summary = f"发现{len(template_clusters)}个相似样本簇，但无显著标签倾斜"
    else:
        summary = "未发现跨样本攻击模板"

    return {
        "total_samples": n,
        "template_clusters": template_clusters,
        "risk_score": round(risk_score, 1),
        "summary": summary,
    }
