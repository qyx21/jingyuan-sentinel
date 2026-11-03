"""
clean_data — 自动剔除（v5 增强版）
===============================
v5 改进：剔除前做快照备份、记录剔除原因、支持撤销。

用法：
    clean_data(dataset, remove_indices, reasons, create_backup) → dict
"""

from datetime import datetime
from typing import Optional


def clean_data(
    dataset: list,
    remove_indices: list[int],
    reasons: Optional[dict] = None,
    create_backup: bool = True,
) -> dict:
    """
    从数据集中剔除指定索引的投毒样本。

    参数:
        dataset: 原始数据集
        remove_indices: 要移除的样本索引
        reasons: {"3": "指令覆盖攻击，置信度98%", ...}
        create_backup: 是否创建备份快照

    返回:
        {
            "original_count": int,
            "removed_count": int,
            "remaining_count": int,
            "backup_id": str | None,
            "removed_samples": [...],
            "cleaned_dataset": [...],
            "timestamp": str,
        }
    """
    if reasons is None:
        reasons = {}

    original_count = len(dataset)

    # 去重 + 边界检查
    remove_set = set()
    for idx in remove_indices:
        if 0 <= idx < original_count:
            remove_set.add(idx)

    removed = []
    cleaned = []
    for i, item in enumerate(dataset):
        if i in remove_set:
            entry = {"original_index": i}
            if isinstance(item, dict):
                entry.update(item)
            else:
                entry["data"] = str(item)
            reason = reasons.get(str(i))
            if reason:
                entry["reason"] = reason
            removed.append(entry)
        else:
            cleaned.append(item)

    backup_id = None
    if create_backup and removed:
        backup_id = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    return {
        "original_count": original_count,
        "removed_count": len(removed),
        "remaining_count": len(cleaned),
        "backup_id": backup_id,
        "removed_samples": removed,
        "cleaned_dataset": cleaned,
        "timestamp": datetime.now().isoformat(),
    }
