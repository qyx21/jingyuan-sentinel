"""
label_consistency_check 测试套件
运行: python tests/test_label_consistency_check.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.stdout.reconfigure(errors="replace")  # 防GBK控制台重定向输出崩溃

from tools.analyzer.label_consistency_check import scan_dataset, _predict_sentiment, _predict_topic


# ═════════════════════════════════════════════════════════
# 情感判定测试
# ═════════════════════════════════════════════════════════

def test_sentiment_negative():
    """明显负面"""
    r = _predict_sentiment("质量太差了，用了三天就坏了，客服也不理人，太坑了")
    assert r["sentiment"] == "negative"
    assert r["confidence"] > 0.5


def test_sentiment_positive():
    """明显正面"""
    r = _predict_sentiment("物流很快包装完好，手机壳手感很好，推荐购买")
    assert r["sentiment"] == "positive"


def test_sentiment_neutral():
    """中性/无情感词"""
    r = _predict_sentiment("今天天气还行")
    assert r["sentiment"] == "neutral"


# ═════════════════════════════════════════════════════════
# 主题判定测试
# ═════════════════════════════════════════════════════════

def test_topic_complaint():
    """投诉主题"""
    r = _predict_topic("我要退款退货，质量有问题，客服态度差")
    assert r == "complaint"


def test_topic_praise():
    """好评主题"""
    r = _predict_topic("强烈推荐，物超所值，一定会回购")
    assert r == "praise"


# ═════════════════════════════════════════════════════════
# 数据集检测测试
# ═════════════════════════════════════════════════════════

def test_empty_dataset():
    """空数据集"""
    r = scan_dataset([])
    assert r["total_samples"] == 0
    assert r["risk_score"] == 0.0


def test_all_consistent():
    """一致的数据集"""
    dataset = [
        {"text": "质量很好很满意推荐购买", "label": "positive"},
        {"text": "物流快包装好手感不错", "label": "positive"},
        {"text": "质量差用了几天就坏了失望", "label": "negative"},
        {"text": "客服态度差不推荐购买", "label": "negative"},
    ]
    r = scan_dataset(dataset)
    # 不应有高危矛盾
    high = [i for i in r["inconsistencies"] if i["severity"] == "high"]
    assert len(high) == 0


def test_sentiment_mismatch():
    """情感标签矛盾"""
    dataset = [
        {"text": "质量非常差用了三天就坏了太失望了客服也不理", "label": "positive"},  # 负面情感，正面标签
        {"text": "物流很快包装好质量不错", "label": "positive"},
    ]
    r = scan_dataset(dataset)
    mismatches = [
        i for i in r["inconsistencies"]
        if any(iss["type"] == "sentiment_label_mismatch" for iss in i["issues"])
    ]
    assert len(mismatches) >= 1


def test_topic_mismatch():
    """主题标签矛盾"""
    dataset = [
        {"text": "我要退款退货质量有问题没人管投诉到底", "label": "positive"},  # 投诉内容，正面标签
        {"text": "物流很快包装完好", "label": "positive"},
    ]
    r = scan_dataset(dataset)
    mismatches = [
        i for i in r["inconsistencies"]
        if any(iss["type"] == "topic_label_mismatch" for iss in i["issues"])
    ]
    assert len(mismatches) >= 1


def test_all_clean_dataset():
    """全正常数据集"""
    dataset = [
        {"text": "物流很快包装完好手机壳手感不错值得购买", "label": "positive"},
        {"text": "这本书内容很丰富适合初学者入门好", "label": "positive"},
        {"text": "咖啡豆新鲜度很好口感醇厚推荐", "label": "positive"},
        {"text": "质量太差了不推荐购买失望退货", "label": "negative"},
        {"text": "客服态度差不理人差评", "label": "negative"},
    ]
    r = scan_dataset(dataset)
    # 正常情况下不应有太多矛盾
    assert r["flagged_count"] < 3


def test_output_structure():
    """输出结构"""
    dataset = [
        {"text": "测试文本", "label": "positive"},
    ]
    r = scan_dataset(dataset)
    assert "total_samples" in r
    assert "inconsistencies" in r
    assert "flagged_indices" in r
    assert "risk_score" in r
    assert "summary" in r
    assert 0 <= r["risk_score"] <= 100


def test_risk_score_range():
    """风险分范围"""
    dataset = [
        {"text": "质量差坏了烂了垃圾", "label": "positive"},
        {"text": "质量差坏了烂了垃圾", "label": "positive"},
        {"text": "质量差坏了烂了垃圾", "label": "positive"},
    ]
    r = scan_dataset(dataset)
    assert 0 <= r["risk_score"] <= 100


def test_no_label_dataset():
    """无标签数据集"""
    dataset = ["文本一", "文本二", "文本三"]
    r = scan_dataset(dataset)
    assert r["total_samples"] == 3
    assert r["flagged_count"] == 0  # 无标签时不应误报


# ═════════════════════════════════════════════════════════
# 运行入口
# ═════════════════════════════════════════════════════════

if __name__ == "__main__":
    sentiment_tests = [
        ("负面情感", test_sentiment_negative),
        ("正面情感", test_sentiment_positive),
        ("中性情感", test_sentiment_neutral),
    ]
    topic_tests = [
        ("投诉主题", test_topic_complaint),
        ("好评主题", test_topic_praise),
    ]
    dataset_tests = [
        ("空数据集", test_empty_dataset),
        ("一致数据", test_all_consistent),
        ("情感矛盾", test_sentiment_mismatch),
        ("主题矛盾", test_topic_mismatch),
        ("全正常", test_all_clean_dataset),
        ("输出结构", test_output_structure),
        ("风险分范围", test_risk_score_range),
        ("无标签", test_no_label_dataset),
    ]

    print("=" * 55)
    print("  label_consistency_check 测试套件")
    print("=" * 55)

    all_suites = [
        ("情感判定（3条）", sentiment_tests),
        ("主题判定（2条）", topic_tests),
        ("数据集检测（8条）", dataset_tests),
    ]

    passed = 0
    failed = 0

    for suite_name, tests in all_suites:
        print(f"\n── {suite_name} ──\n")
        for name, test_fn in tests:
            try:
                test_fn()
                print(f"  ✅ {name}")
                passed += 1
            except AssertionError as e:
                print(f"  ❌ {name}: {e}")
                failed += 1
            except Exception as e:
                print(f"  💥 {name}: {e}")
                failed += 1

    print(f"\n{'=' * 55}")
    print(f"  通过: {passed}  |  失败: {failed}")
    print(f"{'=' * 55}")
