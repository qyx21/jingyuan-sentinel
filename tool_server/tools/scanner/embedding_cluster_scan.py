"""
embedding_cluster_scan — 语义聚类离群检测
=====================================
用 bge-m3 做句子嵌入（768维），HDBSCAN 聚类，检测语义空间中的异常样本。

替代 v4 的 TF-IDF+余弦相似度方案。

检测维度：
1. 噪声点：HDBSCAN 标为 -1，不属于任何簇 → 语义离群 → risk +50
2. 微型簇：成员数 < 总样本的 2% → 可疑投毒群 → risk +40
3. 标签倾斜：微型簇内标签高度一致但偏离全局分布 → 标签投毒 → risk +25
4. 数据集风险：离群率 > 5% → 整体风险升级

调用方式：
    scan_text(text, options)      → 单条（返回None，需要数据集上下文）
    scan_dataset(dataset, options) → 批量检测
"""

import numpy as np
import requests
from typing import Optional
from collections import Counter

# ─── 配置 ────────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434/api/embeddings"
EMBEDDING_MODEL = "bge-m3"
EMBEDDING_DIM = 768
DEFAULT_MIN_CLUSTER_SIZE = 5
DEFAULT_OUTLIER_THRESHOLD = 0.5
DEFAULT_MICRO_CLUSTER_RATIO = 0.02  # <2% 为微型簇


# ═════════════════════════════════════════════════════════
# 嵌入获取
# ═════════════════════════════════════════════════════════

def _get_embeddings(texts: list[str], model: str = EMBEDDING_MODEL) -> Optional[np.ndarray]:
    """
    通过 Ollama 获取 bge-m3 嵌入（4线程并发，单条失败自动跳过）。
    返回 shape (n, 768) 的 numpy 数组，或 None（完全不可达时）。
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    n = len(texts)
    results = [None] * n
    fail_count = 0

    def _fetch_one(text: str, idx: int) -> tuple[int, list[float], bool]:
        try:
            resp = requests.post(
                OLLAMA_URL,
                json={"model": model, "prompt": text},
                timeout=30,
            )
            data = resp.json()
            emb = data.get("embedding")
            if emb is None:
                return idx, None, False
            return idx, emb, True
        except Exception:
            return idx, None, False

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(_fetch_one, text, i): i for i, text in enumerate(texts)}
        for future in as_completed(futures):
            idx, emb, ok = future.result()
            if ok:
                results[idx] = emb
            else:
                fail_count += 1

    if fail_count == n:
        return None  # 全部失败，Ollama 不可达

    # 失败样本用零向量填充，聚类自然标为噪声点
    for i in range(n):
        if results[i] is None:
            results[i] = [0.0] * EMBEDDING_DIM
    return np.array(results, dtype=np.float32)


# ═════════════════════════════════════════════════════════
# PCA 降维（numpy 实现，避免依赖 sklearn 全量安装）
# ═════════════════════════════════════════════════════════

def _pca_reduce(embeddings: np.ndarray, n_components: int = 50) -> np.ndarray:
    """PCA 降维，保留前 n_components 个主成分。"""
    # 中心化
    mean = np.mean(embeddings, axis=0)
    centered = embeddings - mean

    # SVD 分解
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    # s 是从大到小排列的奇异值
    k = min(n_components, len(s), centered.shape[1])
    # 投影到前 k 个主成分
    reduced = centered @ vt[:k].T
    return reduced.astype(np.float32)


# ═════════════════════════════════════════════════════════
# HDBSCAN 简化实现（轻量版，避免依赖 hdbscan 包）
# ═════════════════════════════════════════════════════════
#
# 核心思路：
# 1. 计算互达距离（mutual reachability distance）
# 2. 构建最小生成树
# 3. 按距离切割 → 形成层次聚类
# 4. 提取稳定簇，标记噪声点
#
# 注：完整的 HDBSCAN 算法非常复杂，这里用 DBSCAN + 自适应 eps
#     作为工程近似。通过肘部法则自动选 eps。

def _dbscan_cluster(
    points: np.ndarray,
    min_samples: int = 5,
    eps: Optional[float] = None,
) -> tuple[np.ndarray, float]:
    """
    简化 DBSCAN 聚类（带自适应 eps）。
    返回 (labels, eps_used)
    """
    n = len(points)

    # ── 计算距离矩阵 ──
    # 用批量方式避免 O(n²) 内存爆炸：分批算
    # 对于 n<5000 直接用完整矩阵
    if n <= 5000:
        # 欧几里得距离矩阵
        # ||a-b||² = ||a||² + ||b||² - 2a·b
        sq_norms = np.sum(points ** 2, axis=1)
        dists = np.sqrt(
            np.maximum(sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2 * points @ points.T, 0)
        )
    else:
        raise ValueError(f"数据集过大 ({n} > 5000)，请分批处理")

    # ── 自适应 eps：取第 k 近邻距离的 80 分位数 ──
    if eps is None:
        k = min(min_samples, n - 2) if n > 2 else 1
        if k < 1:
            eps = 1.0
        else:
            k_distances = np.sort(dists, axis=1)[:, k]  # 每行第k近的距离
            if n < 20:
                # 小样本：最大间隔法选 eps（在 k 近邻距离的最大跳变处切开），
                # 避免 80 分位把相距较远的语义组桥接成一个簇
                flat = np.sort(k_distances)
                gaps = np.diff(flat)
                eps = float(flat[int(np.argmax(gaps))]) if len(gaps) > 0 else float(np.percentile(k_distances, 80))
            else:
                eps = float(np.percentile(k_distances, 80))  # 80分位，避免边界点被误判
            if eps < 1e-6:
                eps = 0.1  # 兜底

    # ── DBSCAN ──
    labels = np.full(n, -1, dtype=int)  # -1 = 噪声
    visited = np.zeros(n, dtype=bool)
    cluster_id = 0

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True

        # 找邻居
        neighbors = np.where(dists[i] <= eps)[0]

        if len(neighbors) < min_samples:
            # 噪声点
            continue

        # 扩展簇
        labels[i] = cluster_id
        seeds = list(neighbors[neighbors != i])

        idx = 0
        while idx < len(seeds):
            j = seeds[idx]
            idx += 1

            if visited[j] and labels[j] == -1:
                # 之前标记的噪声点，现在属于这个簇
                labels[j] = cluster_id

            if visited[j]:
                continue
            visited[j] = True

            j_neighbors = np.where(dists[j] <= eps)[0]
            if len(j_neighbors) >= min_samples:
                labels[j] = cluster_id
                # 添加新邻居
                for nb in j_neighbors:
                    if nb not in seeds:
                        seeds.append(nb)
            else:
                labels[j] = cluster_id  # 边界点

        cluster_id += 1

    return labels, eps


# ═════════════════════════════════════════════════════════
# 离群点分析
# ═════════════════════════════════════════════════════════

def _analyze_outliers(
    points: np.ndarray,
    labels: np.ndarray,
    micro_threshold: int,
) -> dict:
    """分析聚类结果，提取离群点和微型簇信息。"""
    n = len(points)
    unique_labels = np.unique(labels)

    # 统计每个簇的大小
    cluster_sizes = {}
    cluster_indices = {}
    for lbl in unique_labels:
        lbl = int(lbl)
        if lbl == -1:
            continue
        mask = labels == lbl
        cluster_sizes[lbl] = int(np.sum(mask))
        cluster_indices[lbl] = [int(i) for i in np.where(mask)[0]]

    # 计算每个簇的中心
    cluster_centers = {}
    for lbl, indices in cluster_indices.items():
        cluster_centers[lbl] = np.mean(points[indices], axis=0)

    # ── 找出微型簇 ──
    micro_clusters = []
    for lbl, size in cluster_sizes.items():
        if size < micro_threshold:
            micro_clusters.append({
                "cluster_id": int(lbl),
                "size": int(size),
                "sample_indices": [int(i) for i in cluster_indices[lbl]],
            })

    # ── 找出噪声点 ──
    noise_mask = labels == -1
    noise_indices = np.where(noise_mask)[0].tolist()

    # ── 对每个异常点，计算到最近正常簇的距离 ──
    outlier_details = []

    # 正常簇中心
    normal_centers = {
        lbl: center for lbl, center in cluster_centers.items()
        if cluster_sizes[lbl] >= micro_threshold
    }

    # 噪声点分析
    for idx in noise_indices:
        pt = points[idx]
        if normal_centers:
            min_dist = float("inf")
            nearest_cluster = -1
            for lbl, center in normal_centers.items():
                d = np.linalg.norm(pt - center)
                if d < min_dist:
                    min_dist = d
                    nearest_cluster = lbl
            outlier_details.append({
                "index": idx,
                "type": "noise",
                "distance_to_nearest_cluster": round(float(min_dist), 3),
                "nearest_cluster_id": nearest_cluster,
            })
        else:
            outlier_details.append({
                "index": idx,
                "type": "noise",
                "distance_to_nearest_cluster": None,
                "nearest_cluster_id": None,
            })

    # 微型簇分析
    for mc in micro_clusters:
        for idx in mc["sample_indices"]:
            pt = points[idx]
            cluster_own_center = cluster_centers.get(mc["cluster_id"])
            if normal_centers and cluster_own_center is not None:
                min_dist = float("inf")
                nearest_cluster = -1
                for lbl, center in normal_centers.items():
                    if lbl != mc["cluster_id"]:
                        d = np.linalg.norm(pt - center)
                        if d < min_dist:
                            min_dist = d
                            nearest_cluster = lbl
                outlier_details.append({
                    "index": idx,
                    "type": "micro_cluster",
                    "cluster_id": mc["cluster_id"],
                    "cluster_size": mc["size"],
                    "distance_to_nearest_normal_cluster": round(float(min_dist), 3),
                })
            else:
                outlier_details.append({
                    "index": idx,
                    "type": "micro_cluster",
                    "cluster_id": mc["cluster_id"],
                    "cluster_size": mc["size"],
                    "distance_to_nearest_normal_cluster": None,
                })

    # 去重（按 index）
    seen = set()
    unique_outliers = []
    for od in outlier_details:
        if od["index"] not in seen:
            unique_outliers.append(od)
            seen.add(od["index"])

    return {
        "num_clusters": len([l for l in unique_labels if l != -1]),
        "cluster_sizes": {int(k): v for k, v in cluster_sizes.items()},
        "noise_count": len(noise_indices),
        "noise_indices": noise_indices,
        "micro_clusters": micro_clusters,
        "outlier_details": sorted(unique_outliers, key=lambda x: x["index"]),
        "eps_used": None,  # 将在主函数中设置
    }


# ═════════════════════════════════════════════════════════
# 标签倾斜分析
# ═════════════════════════════════════════════════════════

def _analyze_label_skew(
    labels: np.ndarray,
    dataset: list,
) -> dict:
    """
    检测微型簇内的标签倾斜：
    - 如果某个簇内标签高度一致，但与全局标签分布不同 → 标签投毒嫌疑
    """
    # 提取全局标签分布
    global_labels = []
    for item in dataset:
        if isinstance(item, dict):
            global_labels.append(item.get("label", "unknown"))
        else:
            global_labels.append("unknown")

    global_dist = Counter(global_labels)
    total = len(global_labels)

    unique_labels = np.unique(labels)
    skewed_clusters = []

    for lbl in unique_labels:
        if lbl == -1:
            continue
        mask = labels == lbl
        cluster_global_labels = [global_labels[i] for i in range(len(global_labels)) if mask[i]]
        if not cluster_global_labels:
            continue

        cluster_dist = Counter(cluster_global_labels)
        cluster_total = len(cluster_global_labels)

        # 计算簇内标签一致性（最高标签占比）
        if cluster_total > 0:
            max_label, max_count = cluster_dist.most_common(1)[0]
            consistency = max_count / cluster_total

            # 该标签在全局的占比
            global_ratio = global_dist.get(max_label, 0) / max(total, 1)

            # 如果簇内高度一致（>0.85）但偏离全局分布（差值>0.3）→ 标签倾斜
            if consistency > 0.85 and abs(consistency - global_ratio) > 0.3:
                skewed_clusters.append({
                    "cluster_id": int(lbl),
                    "dominant_label": max_label,
                    "cluster_consistency": round(consistency, 3),
                    "global_ratio": round(global_ratio, 3),
                    "skew": round(consistency - global_ratio, 3),
                    "sample_indices": np.where(mask)[0].tolist(),
                })

    return {
        "has_skew": len(skewed_clusters) > 0,
        "skewed_clusters": skewed_clusters,
    }


# ═════════════════════════════════════════════════════════
# 公开 API
# ═════════════════════════════════════════════════════════

def scan_dataset(dataset: list, options: Optional[dict] = None) -> dict:
    """
    批量语义聚类离群检测。

    参数:
        dataset: [{"text": "...", "label": "..."}, ...] 或 ["text1", "text2", ...]
        options: {
            "model": "bge-m3",
            "min_cluster_size": 5,
            "outlier_threshold": 0.5,
            "micro_cluster_ratio": 0.02,
        }

    返回:
        {
            "total_samples": int,
            "num_clusters": int,
            "noise_points": int,
            "micro_clusters": [...],
            "outliers": [...],
            "label_skew": {...},
            "flagged_indices": [...],
            "overall_risk_score": 0-100,
        }
    """
    if options is None:
        options = {}

    model = options.get("model", EMBEDDING_MODEL)
    min_cluster_size = options.get("min_cluster_size", DEFAULT_MIN_CLUSTER_SIZE)
    micro_cluster_ratio = options.get("micro_cluster_ratio", DEFAULT_MICRO_CLUSTER_RATIO)

    # 统一为文本列表
    texts = []
    for item in dataset:
        if isinstance(item, str):
            texts.append(item)
        elif isinstance(item, dict):
            texts.append(item.get("text", item.get("content", str(item))))
        else:
            texts.append(str(item))

    n = len(texts)

    if n < 3:
        return {
            "total_samples": n,
            "num_clusters": 0,
            "noise_points": 0,
            "micro_clusters": [],
            "outliers": [],
            "label_skew": {"has_skew": False, "skewed_clusters": []},
            "flagged_indices": [],
            "overall_risk_score": 0.0,
            "error": "样本数不足（需要≥3），跳过聚类",
        }

    # 小数据集自适应：默认簇下限(5)对 <20 条的数据过严，收缩到 n//3
    if n < 20:
        min_cluster_size = min(min_cluster_size, max(2, n // 3))

    # ── 1. 获取嵌入 ──
    try:
        embeddings = _get_embeddings(texts, model)
    except Exception as e:
        return {
            "total_samples": n,
            "error": f"嵌入获取失败: {e}",
            "overall_risk_score": 0.0,
        }

    if embeddings is None:
        return {
            "total_samples": n,
            "error": "Ollama 服务不可达，请确认 bge-m3 已部署",
            "overall_risk_score": 0.0,
        }

    # ── 2. PCA 降维 ──
    n_components = min(50, n - 1, embeddings.shape[1])
    if n_components > 1:
        reduced = _pca_reduce(embeddings, n_components=n_components)
    else:
        reduced = embeddings

    # ── 3. 聚类 ──
    cluster_labels, eps_used = _dbscan_cluster(
        reduced,
        min_samples=min_cluster_size,
        eps=None,  # 自适应
    )

    # ── 4. 异常分析 ──
    micro_threshold = max(min_cluster_size, int(n * micro_cluster_ratio))
    analysis = _analyze_outliers(reduced, cluster_labels, micro_threshold)
    analysis["eps_used"] = round(float(eps_used), 4)

    # ── 5. 标签倾斜检测 ──
    label_skew = _analyze_label_skew(cluster_labels, dataset)

    # ── 6. 计算风险分 ──
    flagged_set = set()
    per_sample_risk = [0] * n

    # 噪声点
    for od in analysis["outlier_details"]:
        idx = od["index"]
        if od["type"] == "noise":
            per_sample_risk[idx] += 50
            flagged_set.add(idx)

    # 微型簇
    for mc in analysis["micro_clusters"]:
        for idx in mc["sample_indices"]:
            per_sample_risk[idx] += 40
            flagged_set.add(idx)

    # 标签倾斜
    for sc in label_skew["skewed_clusters"]:
        for idx in sc["sample_indices"]:
            per_sample_risk[idx] += 25
            flagged_set.add(idx)

    # 上限 100
    per_sample_risk = [min(100, r) for r in per_sample_risk]
    flagged_indices = sorted(list(flagged_set))

    anomaly_ratio = len(flagged_indices) / max(n, 1)
    overall_risk = round(np.mean(per_sample_risk) if per_sample_risk else 0, 1)
    if anomaly_ratio > 0.05:
        overall_risk = min(100, overall_risk + 20)

    # ── 7. 构建输出 ──
    # 构建 outlier 列表（含最近邻信息）
    outliers_formatted = []
    for od in analysis["outlier_details"]:
        idx = od["index"]
        entry = {
            "index": idx,
            "type": od["type"],
            "text_preview": texts[idx][:100],
        }
        if od["type"] == "noise":
            entry["distance_to_nearest_cluster"] = od.get("distance_to_nearest_cluster")
        else:
            entry["cluster_id"] = od.get("cluster_id")
            entry["cluster_size"] = od.get("cluster_size")
            entry["distance_to_nearest_normal_cluster"] = od.get("distance_to_nearest_normal_cluster")
        outliers_formatted.append(entry)

    # 微型簇输出
    micro_clusters_formatted = []
    for mc in analysis["micro_clusters"]:
        micro_clusters_formatted.append({
            "cluster_id": mc["cluster_id"],
            "size": mc["size"],
            "sample_indices": mc["sample_indices"],
            "representative_samples": [texts[i][:80] for i in mc["sample_indices"][:5]],
        })

    return {
        "total_samples": n,
        "num_clusters": analysis["num_clusters"],
        "noise_points": analysis["noise_count"],
        "micro_clusters": micro_clusters_formatted,
        "outliers": outliers_formatted,
        "label_skew": label_skew,
        "flagged_count": len(flagged_indices),
        "flagged_indices": flagged_indices,
        "anomaly_ratio": round(anomaly_ratio, 3),
        "per_sample_risk": per_sample_risk,
        "overall_risk_score": round(overall_risk, 1),
        "parameters": {
            "model": model,
            "min_cluster_size": min_cluster_size,
            "eps_auto": analysis["eps_used"],
            "micro_cluster_threshold": micro_threshold,
            "pca_components": n_components,
        },
    }
