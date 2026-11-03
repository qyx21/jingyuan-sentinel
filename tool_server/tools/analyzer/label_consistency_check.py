"""
label_consistency_check — 标签一致性检测
=======================================
检测训练数据中"内容与标签矛盾"的样本——标签投毒（label poisoning）的核心特征。

检测维度：
1. 情感-标签矛盾：负面情感但标为 positive（反之亦然）
2. 主题-标签矛盾：投诉/退款内容但标为 positive
3. 长度-标签矛盾：文本长度偏离同标签分布
4. 关键词-标签矛盾：标签下出现不应有的关键词

实现：基于关键词词典的情感判定（无需额外模型下载，中文电商场景有效）。

调用方式：
    scan_dataset(dataset, options) → dict
"""

import re
import numpy as np
from collections import Counter
from typing import Optional


# ═════════════════════════════════════════════════════════
# 中文情感关键词词典
# ═════════════════════════════════════════════════════════

NEGATIVE_KEYWORDS = {
    # 强烈负面
    "差", "烂", "垃圾", "坑", "骗", "假", "劣质", "次品", "失望",
    "糟", "坏", "破", "臭", "脏",
    # 质量负面
    "坏了", "破损", "摔坏", "断裂", "掉色", "脱线", "开胶", "变形",
    "生锈", "发霉", "异味", "粗糙", "噪音", "卡顿", "死机", "闪退",
    # 体验负面
    "难用", "不好用", "不方便", "不实用", "不划算", "不值",
    "后悔", "退", "投诉", "差评", "举报", "维权",
    "慢", "太慢", "拖延", "迟迟", "延迟",
    # 英文负面
    "terrible", "horrible", "awful", "worst", "garbage", "trash",
    "waste", "useless", "broken", "defective", "damaged", "poor",
    "disappointed", "disappointing", "scam", "fake", "cheap crap",
    "do not buy", "don't buy", "never buy", "avoid",
    # 中文隐式负面
    "白花", "浪费钱", "上当了", "被坑了", "心累", "无语",
    "无语了", "服了", "醉了", "吐血", "坑爹", "扯淡",
}

POSITIVE_KEYWORDS = {
    # 强烈正面
    "好", "棒", "赞", "优秀", "出色", "完美", "满意", "推荐",
    "超值", "划算", "物美价廉",
    # 质量正面
    "质量好", "质量不错", "做工好", "精致", "耐用", "结实",
    "手感好", "柔软", "舒服", "舒适", "轻薄", "顺滑",
    # 体验正面
    "好用", "实用", "方便", "快", "很快", "迅速", "高效",
    "良心", "靠谱", "值得", "回购", "好评",
    # 服务正面
    "服务好", "态度好", "热情", "周到", "耐心",
    # 英文正面
    "excellent", "great", "amazing", "awesome", "perfect",
    "fantastic", "wonderful", "love", "recommend", "best",
    "worth it", "good quality", "happy", "satisfied",
    "fast", "durable", "comfortable", "impressive",
}

# 投诉/退款 主题词
COMPLAINT_KEYWORDS = {
    "退款", "退货", "换货", "投诉", "维权", "赔偿", "赔付",
    "质量问题", "虚假宣传", "货不对板", "假货", "仿冒",
    "客服不理", "客服态度差", "售后差", "没人管",
    "refund", "return", "complaint", "compensation", "dispute",
    "chargeback", "fake product", "counterfeit",
}

# 好评/推荐 主题词
PRAISE_KEYWORDS = {
    "推荐", "好评", "值得买", "物超所值", "必入", "种草",
    "回购", "回头客", "老顾客", "忠实粉丝",
    "recommend", "must buy", "repurchase", "five stars",
    "5 stars", "10/10", "favorite",
}


# ═════════════════════════════════════════════════════════
# 情感判定
# ═════════════════════════════════════════════════════════

# 否定负面词 = 正面含义（"没有坏""没问题"等），直接剥离避免误判
NEGATED_NEGATIVES = ["没有坏", "没坏", "不坏", "没问题", "没有色差", "没有异味", "不难用", "不慢"]


def _predict_sentiment(text: str) -> dict:
    """
    基于关键词词典预测文本情感。
    返回 {"sentiment": "positive"|"negative"|"neutral", "confidence": 0-1, "evidence": [...]}
    """
    text_lower = text.lower()
    for negated in NEGATED_NEGATIVES:
        text_lower = text_lower.replace(negated, "")

    neg_count = 0
    pos_count = 0
    neg_evidence = []
    pos_evidence = []

    for kw in NEGATIVE_KEYWORDS:
        if kw in text_lower:
            neg_count += 1
            neg_evidence.append(kw)
            if neg_count >= 5:
                break  # 足够了

    for kw in POSITIVE_KEYWORDS:
        if kw in text_lower:
            pos_count += 1
            pos_evidence.append(kw)
            if pos_count >= 5:
                break

    # 计算置信度
    total = neg_count + pos_count
    if total == 0:
        return {"sentiment": "neutral", "confidence": 0.0, "evidence": []}

    # 纯一边倒 → 高置信度
    if neg_count > 0 and pos_count == 0:
        confidence = min(1.0, neg_count / 4)
        return {"sentiment": "negative", "confidence": round(confidence, 2), "evidence": neg_evidence[:5]}
    elif pos_count > 0 and neg_count == 0:
        confidence = min(1.0, pos_count / 4)
        return {"sentiment": "positive", "confidence": round(confidence, 2), "evidence": pos_evidence[:5]}

    # 混合 → 看比例
    ratio = neg_count / total
    if ratio > 0.7:
        confidence = min(1.0, ratio)
        return {"sentiment": "negative", "confidence": round(confidence, 2), "evidence": neg_evidence[:5]}
    elif ratio < 0.3:
        confidence = min(1.0, 1.0 - ratio)
        return {"sentiment": "positive", "confidence": round(confidence, 2), "evidence": pos_evidence[:5]}

    return {"sentiment": "neutral", "confidence": 0.0, "evidence": []}


# ═════════════════════════════════════════════════════════
# 主题判定
# ═════════════════════════════════════════════════════════

def _predict_topic(text: str) -> str:
    """基于关键词判定文本主题倾向。"""
    # 否定处理："不推荐"不是好评信号
    text_lower = text.lower().replace("不推荐", "")

    complaint_hits = sum(1 for kw in COMPLAINT_KEYWORDS if kw in text_lower)
    praise_hits = sum(1 for kw in PRAISE_KEYWORDS if kw in text_lower)

    if complaint_hits > 0 and praise_hits == 0:
        return "complaint"
    elif praise_hits > 0 and complaint_hits == 0:
        return "praise"
    elif complaint_hits > praise_hits:
        return "complaint"
    elif praise_hits > complaint_hits:
        return "praise"
    return "neutral"


# ═════════════════════════════════════════════════════════
# 公开 API
# ═════════════════════════════════════════════════════════

def scan_dataset(dataset: list, options: Optional[dict] = None) -> dict:
    """
    批量标签一致性检测。

    参数:
        dataset: [{"text": "...", "label": "positive"}, ...] 或 ["text", ...]
        options: {
            "check_sentiment": True,
            "check_topic": True,
            "check_length": True,
            "check_keywords": True,
        }

    返回:
        {
            "total_samples": int,
            "inconsistencies": [...],
            "risk_score": 0-100,
            "summary": str,
        }
    """
    if options is None:
        options = {}

    check_sentiment = options.get("check_sentiment", True)
    check_topic = options.get("check_topic", True)
    check_length = options.get("check_length", True)
    check_keywords = options.get("check_keywords", True)

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

    if n == 0:
        return {
            "total_samples": 0,
            "inconsistencies": [],
            "risk_score": 0.0,
            "summary": "数据集为空",
        }

    # ── 长度分布（按标签） ──
    length_by_label = {}
    for text, label in zip(texts, labels):
        if label not in length_by_label:
            length_by_label[label] = []
        length_by_label[label].append(len(text))

    label_length_stats = {}
    for lbl, lengths in length_by_label.items():
        if len(lengths) >= 2:
            label_length_stats[lbl] = {
                "mean": float(np.mean(lengths)),
                "std": float(np.std(lengths)),
                "count": len(lengths),
            }
        else:
            label_length_stats[lbl] = {
                "mean": float(np.mean(lengths)) if lengths else 0,
                "std": 0.0,
                "count": len(lengths),
            }

    # ── 标签关键词统计 ──
    label_kw_stats = {}
    for lbl in set(labels):
        label_texts = [texts[i] for i in range(n) if labels[i] == lbl]
        all_text = ' '.join(label_texts).lower()
        neg_in_label = sum(1 for kw in NEGATIVE_KEYWORDS if kw in all_text)
        pos_in_label = sum(1 for kw in POSITIVE_KEYWORDS if kw in all_text)
        label_kw_stats[lbl] = {
            "negative_keyword_count": neg_in_label,
            "positive_keyword_count": pos_in_label,
        }

    # ── 逐条检测矛盾 ──
    inconsistencies = []

    for i in range(n):
        text = texts[i]
        label = labels[i]
        issues = []
        severity_score = 0

        # 1) 情感-标签矛盾
        if check_sentiment and label != "unknown":
            sentiment = _predict_sentiment(text)
            if sentiment["sentiment"] != "neutral" and sentiment["confidence"] >= 0.25:
                if sentiment["sentiment"] == "negative" and label.lower() in ("positive", "pos", "1", "好评"):
                    issues.append({
                        "type": "sentiment_label_mismatch",
                        "detail": f"情感负面但标签为{label}",
                        "predicted_sentiment": "negative",
                        "confidence": sentiment["confidence"],
                    })
                    severity_score += 40
                elif sentiment["sentiment"] == "positive" and label.lower() in ("negative", "neg", "0", "差评"):
                    issues.append({
                        "type": "sentiment_label_mismatch",
                        "detail": f"情感正面但标签为{label}",
                        "predicted_sentiment": "positive",
                        "confidence": sentiment["confidence"],
                    })
                    severity_score += 40

        # 2) 主题-标签矛盾
        if check_topic and label != "unknown":
            topic = _predict_topic(text)
            if topic == "complaint" and label.lower() in ("positive", "pos", "1", "好评"):
                issues.append({
                    "type": "topic_label_mismatch",
                    "detail": f"投诉/退款内容但标签为{label}",
                    "predicted_topic": "complaint",
                })
                severity_score += 35
            elif topic == "praise" and label.lower() in ("negative", "neg", "0", "差评"):
                issues.append({
                    "type": "topic_label_mismatch",
                    "detail": f"好评/推荐内容但标签为{label}",
                    "predicted_topic": "praise",
                })
                severity_score += 35

        # 3) 长度-标签矛盾
        if check_length and label in label_length_stats:
            stats = label_length_stats[label]
            if stats["std"] > 0 and stats["count"] >= 3:
                z_score = (len(text) - stats["mean"]) / max(stats["std"], 1)
                if abs(z_score) > 3:
                    issues.append({
                        "type": "length_anomaly",
                        "detail": f"长度{len(text)}字，偏离{label}标签均值{stats['mean']:.0f}字 (z={z_score:.1f})",
                    })
                    severity_score += 15

        # 4) 关键词-标签矛盾
        if check_keywords and label != "unknown":
            sentiment = _predict_sentiment(text)
            neg_evidence = [kw for kw in NEGATIVE_KEYWORDS if kw in text.lower()]
            pos_evidence = [kw for kw in POSITIVE_KEYWORDS if kw in text.lower()]

            if label.lower() in ("positive", "pos", "1", "好评"):
                if len(neg_evidence) >= 3 and len(pos_evidence) == 0:
                    issues.append({
                        "type": "keyword_label_mismatch",
                        "detail": f"positive标签下出现大量负面词",
                        "evidence_keywords": neg_evidence[:5],
                    })
                    severity_score += 25
            elif label.lower() in ("negative", "neg", "0", "差评"):
                if len(pos_evidence) >= 3 and len(neg_evidence) == 0:
                    issues.append({
                        "type": "keyword_label_mismatch",
                        "detail": f"negative标签下出现大量正面词",
                        "evidence_keywords": pos_evidence[:5],
                    })
                    severity_score += 25

        if issues:
            # 严重程度
            if severity_score >= 70:
                severity = "high"
            elif severity_score >= 35:
                severity = "medium"
            else:
                severity = "low"

            # 提取证据关键词
            evidence_keywords = []
            for iss in issues:
                if "evidence_keywords" in iss:
                    evidence_keywords.extend(iss["evidence_keywords"])

            inconsistencies.append({
                "index": i,
                "text_preview": text[:100],
                "given_label": label,
                "issues": issues,
                "evidence_keywords": list(set(evidence_keywords))[:10],
                "severity": severity,
                "risk_score": min(100, severity_score),
            })

    # ── 风险评分 ──
    if not inconsistencies:
        risk_score = 0.0
    else:
        avg_severity = sum(inc["risk_score"] for inc in inconsistencies) / max(len(inconsistencies), 1)
        ratio = len(inconsistencies) / max(n, 1)
        risk_score = min(avg_severity * min(ratio * 10, 2), 100.0)

    flagged_indices = [inc["index"] for inc in inconsistencies if inc["severity"] in ("high", "medium")]

    # ── 汇总 ──
    high_n = sum(1 for inc in inconsistencies if inc["severity"] == "high")
    medium_n = sum(1 for inc in inconsistencies if inc["severity"] == "medium")
    sentiment_n = sum(
        1 for inc in inconsistencies
        if any(iss["type"] == "sentiment_label_mismatch" for iss in inc["issues"])
    )

    if inconsistencies:
        summary = (
            f"发现{len(inconsistencies)}条标签矛盾"
            f"（高危{high_n}，中危{medium_n}），"
            f"其中情感矛盾{sentiment_n}条"
        )
    else:
        summary = "未发现标签一致性异常"

    return {
        "total_samples": n,
        "inconsistencies": inconsistencies,
        "flagged_count": len(flagged_indices),
        "flagged_indices": flagged_indices,
        "risk_score": round(risk_score, 1),
        "summary": summary,
    }
