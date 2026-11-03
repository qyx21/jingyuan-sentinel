"""
quarantine_data — 隔离存疑样本
===============================
对无法确定的中危样本，不直接删除（避免误删有价值数据），而是隔离到待审区。

与 clean_data 的区别：
  - clean_data：确定是毒，直接删
  - quarantine：不确定，隔离等人工复核

用法：
    quarantine_data(dataset, quarantine_indices, reasons) → dict
"""

from datetime import datetime
from typing import Optional


def quarantine_data(
    dataset: list,
    quarantine_indices: list[int],
    reasons: Optional[dict] = None,
) -> dict:
    """
    隔离存疑样本到待审区。

    参数:
        dataset: 原始数据集
        quarantine_indices: 要隔离的样本索引
        reasons: {"67": "仅1个检测器标记，疑似新攻击类型", ...}

    返回:
        {
            "original_count": int,
            "quarantined_count": int,
            "active_count": int,
            "quarantined_samples": [...],
            "active_dataset": [...],
            "review_required": bool,
            "review_deadline_suggestion": str,
            "timestamp": str,
        }
    """
    if reasons is None:
        reasons = {}

    original_count = len(dataset)

    # 去重 + 边界检查
    quarantine_set = set()
    for idx in quarantine_indices:
        if 0 <= idx < original_count:
            quarantine_set.add(idx)

    quarantined = []
    active = []
    for i, item in enumerate(dataset):
        if i in quarantine_set:
            entry = {"original_index": i}
            if isinstance(item, dict):
                entry.update(item)
            else:
                entry["data"] = str(item)
            reason = reasons.get(str(i))
            if reason:
                entry["reason"] = reason
            quarantined.append(entry)
        else:
            active.append(item)

    review_required = len(quarantined) > 0

    return {
        "original_count": original_count,
        "quarantined_count": len(quarantined),
        "active_count": len(active),
        "quarantined_samples": quarantined,
        "active_dataset": active,
        "review_required": review_required,
        "review_deadline_suggestion": (
            "建议在训练前完成人工复核" if review_required else ""
        ),
        "timestamp": datetime.now().isoformat(),
    }
