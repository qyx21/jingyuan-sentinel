"""
verify_clean — 清洗后验证（v5 增强版）
=====================================
三层验证：完整性 → 残留检测 → 分布健康

v5 改进：不是简单重跑扫描，而是分层次验证。

用法：
    verify_clean(dataset, removed_indices, original_stats, options) → dict
"""

import numpy as np
from collections import Counter


def verify_clean(
    dataset: list,
    removed_indices: list[int] = None,
    original_stats: dict = None,
    options: dict = None,
) -> dict:
    """
    三层清洗后验证。

    参数:
        dataset: 清洗后的数据集
        removed_indices: 被移除的样本索引
        original_stats: {"total": 1000, "label_distribution": {...}, "mean_text_length": 85}
        options: {"run_residual_scan": True}

    返回:
        {
            "status": "clean"|"dirty"|"warning",
            "integrity_check": {...},
            "residual_scan": {...},
            "distribution_health": {...},
            "message": str,
        }
    """
    if removed_indices is None:
        removed_indices = []
    if options is None:
        options = {}
    if original_stats is None:
        original_stats = {}

    run_residual = options.get("run_residual_scan", True)
    remaining_count = len(dataset)

    # ── 层次1：完整性检查 ──
    orig_total = original_stats.get("total", remaining_count + len(removed_indices))
    expected_remaining = orig_total - len(removed_indices)
    integrity_ok = (remaining_count == expected_remaining)

    integrity = {
        "passed": integrity_ok,
        "removed_verified": len(removed_indices),
        "remaining_verified": remaining_count,
        "expected_remaining": expected_remaining,
        "detail": (
            "完整性检查通过" if integrity_ok
            else f"数量不匹配：预期{expected_remaining}，实际{remaining_count}"
        ),
    }

    # ── 层次2：残留检测（快速模式） ──
    residual = {}
    if run_residual and remaining_count > 0:
        try:
            from ..scanner.unicode_deep_scan import scan_text as unicode_scan
            from ..analyzer.instruction_detect import scan_text as detect_instruction

            high_risk_found = 0
            medium_risk_found = 0
            details = []

            for i, item in enumerate(dataset):
                text = item.get("text", str(item)) if isinstance(item, dict) else str(item)

                uresult = unicode_scan(text)
                if uresult["risk_score"] >= 60:
                    high_risk_found += 1
                    details.append({
                        "index": i, "source": "unicode_scan",
                        "risk_score": uresult["risk_score"], "text_preview": text[:80],
                    })
                    continue

                iresult = detect_instruction(text)
                if iresult["severity"] in ("high", "medium"):
                    if iresult["severity"] == "high":
                        high_risk_found += 1
                    else:
                        medium_risk_found += 1
                    details.append({
                        "index": i, "source": "instruction_detect",
                        "severity": iresult["severity"], "text_preview": text[:80],
                    })

            residual = {
                "new_high_risk_found": high_risk_found,
                "new_medium_risk_found": medium_risk_found,
                "details": details[:10],
            }
        except Exception as e:
            residual = {"error": f"残留扫描异常: {e}", "new_high_risk_found": 0, "new_medium_risk_found": 0, "details": []}

    # ── 层次3：分布健康检查 ──
    distribution = {}
    if original_stats:
        orig_label_dist = original_stats.get("label_distribution", {})
        label_drift = False
        drift_details = []

        if orig_label_dist:
            curr_labels = [
                item.get("label", "unknown") if isinstance(item, dict) else "unknown"
                for item in dataset
            ]
            curr_label_dist = dict(Counter(curr_labels))
            orig_sum = sum(orig_label_dist.values())
            curr_sum = len(curr_labels)

            for lbl in set(list(orig_label_dist.keys()) + list(curr_label_dist.keys())):
                orig_r = orig_label_dist.get(lbl, 0) / max(orig_sum, 1)
                curr_r = curr_label_dist.get(lbl, 0) / max(curr_sum, 1)
                drift = abs(curr_r - orig_r)
                if drift > 0.15:
                    label_drift = True
                    drift_details.append({
                        "label": lbl, "original_ratio": round(orig_r, 3),
                        "current_ratio": round(curr_r, 3), "drift": round(drift, 3),
                    })

        orig_mean_len = original_stats.get("mean_text_length", 0)
        length_shift = 0.0
        if orig_mean_len > 0 and remaining_count > 0:
            curr_lengths = [
                len(item.get("text", str(item))) if isinstance(item, dict) else len(str(item))
                for item in dataset
            ]
            curr_mean = np.mean(curr_lengths) if curr_lengths else 0
            length_shift = round(abs(curr_mean - orig_mean_len) / max(orig_mean_len, 1), 3)

        distribution = {
            "label_drift_detected": label_drift,
            "drift_details": drift_details,
            "length_distribution_shift": length_shift,
            "length_healthy": length_shift < 0.1,
            "warning": "标签分布出现显著漂移，可能误删了正常样本" if label_drift else None,
        }

    # ── 综合判定 ──
    high_new = residual.get("new_high_risk_found", 0)
    mid_new = residual.get("new_medium_risk_found", 0)
    label_drift = distribution.get("label_drift_detected", False)

    if not integrity_ok:
        status = "dirty"
        message = f"⚠ 完整性检查失败，数据可能未正确清洗"
    elif high_new > 0:
        status = "dirty"
        message = f"⚠ 残留扫描发现 {high_new} 条高风险样本，建议再次清洗"
    elif label_drift:
        status = "warning"
        message = "⚠ 标签分布出现漂移，建议复核剔除列表"
    elif mid_new > 0:
        status = "warning"
        message = f"⚠ 残留 {mid_new} 条中风险样本，可考虑隔离审查"
    else:
        status = "clean"
        message = f"✓ 验证通过。{len(removed_indices)}条投毒已移除，{remaining_count}条数据可安全用于训练"

    return {
        "status": status,
        "integrity_check": integrity,
        "residual_scan": residual,
        "distribution_health": distribution,
        "message": message,
    }
