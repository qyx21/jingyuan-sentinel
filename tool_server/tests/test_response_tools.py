"""
响应+报告工具测试套件
运行: python tests/test_response_tools.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.stdout.reconfigure(errors="replace")  # 防GBK控制台重定向输出崩溃

from tools.response.clean_data import clean_data
from tools.response.quarantine_data import quarantine_data
from tools.response.verify_clean import verify_clean
from tools.report.generate_report import generate_report


# ═════════════════════════════════════════════════════════
# clean_data 测试
# ═════════════════════════════════════════════════════════

def test_clean_basic():
    """基本剔除"""
    dataset = [
        {"text": "文本一", "label": "positive"},
        {"text": "文本二", "label": "positive"},
        {"text": "文本三", "label": "negative"},
    ]
    r = clean_data(dataset, [0, 2])
    assert r["original_count"] == 3
    assert r["removed_count"] == 2
    assert r["remaining_count"] == 1
    assert r["cleaned_dataset"][0]["text"] == "文本二"


def test_clean_with_reasons():
    """带原因的剔除"""
    dataset = [{"text": f"text{i}", "label": "pos"} for i in range(5)]
    reasons = {"0": "指令覆盖攻击", "4": "后门触发词"}
    r = clean_data(dataset, [0, 4], reasons=reasons)
    assert r["removed_samples"][0]["reason"] == "指令覆盖攻击"
    assert r["backup_id"] is not None


def test_clean_empty():
    """空剔除"""
    r = clean_data([], [])
    assert r["original_count"] == 0


def test_clean_out_of_bounds():
    """越界索引"""
    dataset = [{"text": "a"}, {"text": "b"}]
    r = clean_data(dataset, [0, 99, -1])
    assert r["removed_count"] == 1  # 只有索引0有效


# ═════════════════════════════════════════════════════════
# quarantine_data 测试
# ═════════════════════════════════════════════════════════

def test_quarantine_basic():
    """基本隔离"""
    dataset = [
        {"text": "正常文本一", "label": "positive"},
        {"text": "可疑文本二", "label": "positive"},
        {"text": "正常文本三", "label": "negative"},
    ]
    r = quarantine_data(dataset, [1], reasons={"1": "疑似新攻击类型"})
    assert r["original_count"] == 3
    assert r["quarantined_count"] == 1
    assert r["active_count"] == 2
    assert r["review_required"] is True


def test_quarantine_none():
    """无需隔离"""
    r = quarantine_data([{"text": "a"}], [])
    assert r["quarantined_count"] == 0
    assert r["review_required"] is False


# ═════════════════════════════════════════════════════════
# verify_clean 测试
# ═════════════════════════════════════════════════════════

def test_verify_basic():
    """基本验证"""
    dataset = [
        {"text": "物流很快包装完好手机壳手感不错", "label": "positive"},
        {"text": "这本书内容丰富适合初学者", "label": "positive"},
        {"text": "咖啡豆新鲜度很好口感醇厚", "label": "positive"},
    ]
    r = verify_clean(dataset, [45], {"total": 53, "mean_text_length": 12})
    assert r["status"] in ("clean", "warning", "dirty")
    assert "integrity_check" in r
    assert "distribution_health" in r
    assert "message" in r


def test_verify_empty():
    """空数据集验证"""
    r = verify_clean([], [], {"total": 0, "mean_text_length": 0})
    assert r["integrity_check"]["passed"] is True


def test_verify_label_drift():
    """标签漂移检测"""
    dataset = [
        {"text": "测试文本", "label": "negative"},
        {"text": "测试文本", "label": "negative"},
    ]
    r = verify_clean(dataset, [0], {
        "total": 3,
        "label_distribution": {"positive": 2, "negative": 1},
        "mean_text_length": 4,
    })
    assert "distribution_health" in r


# ═════════════════════════════════════════════════════════
# generate_report 测试
# ═════════════════════════════════════════════════════════

def test_report_markdown():
    """Markdown 报告"""
    r = generate_report({
        "session_id": "test-001",
        "total_samples": 100,
        "full_scan": {
            "flagged_indices": [1, 2],
            "aggregated_risk_score": 35.0,
            "scanner_results": {
                "unicode": {"overall_risk_score": 20, "total_flags": 1},
                "perplexity": {"overall_risk_score": 30, "flagged_count": 2, "statistics": {"naturalness_mean": 7.5, "naturalness_std": 1.2}},
                "embedding": {"overall_risk_score": 25, "num_clusters": 3, "noise_points": 2, "micro_clusters": []},
            },
            "cross_validated_flags": [{"confidence": "high", "index": 1}],
        },
        "trigger_analysis": {"suspicious_triggers": []},
        "instruction_detect": {"statistics": {"confirmed_attacks": 1, "discussions": 0, "variants": 0}},
        "label_consistency_check": {"inconsistencies": []},
        "cross_sample_correlation": {"template_clusters": []},
        "disposition": {"cleaned": [], "quarantined": []},
        "verification": {"status": "clean"},
    })
    assert "净源哨兵" in r["report"]
    assert "test-001" in r["report"]
    assert "执行摘要" in r["report"]


def test_report_json():
    """JSON 格式报告"""
    r = generate_report({"session_id": "t1", "total_samples": 10}, fmt="json")
    assert r["format"] == "json"
    assert "t1" in r["report"]


# ═════════════════════════════════════════════════════════
# 运行入口
# ═════════════════════════════════════════════════════════

if __name__ == "__main__":
    all_tests = [
        ("剔除-基本", test_clean_basic),
        ("剔除-带原因", test_clean_with_reasons),
        ("剔除-空", test_clean_empty),
        ("剔除-越界", test_clean_out_of_bounds),
        ("隔离-基本", test_quarantine_basic),
        ("隔离-无需", test_quarantine_none),
        ("验证-基本", test_verify_basic),
        ("验证-空", test_verify_empty),
        ("验证-标签漂移", test_verify_label_drift),
        ("报告-Markdown", test_report_markdown),
        ("报告-JSON", test_report_json),
    ]

    print("=" * 55)
    print("  响应+报告工具 测试套件")
    print("=" * 55)

    passed = 0
    failed = 0

    for name, test_fn in all_tests:
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
