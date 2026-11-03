"""
知识库工具测试套件
运行: python tests/test_knowledge_tools.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.stdout.reconfigure(errors="replace")  # 防GBK控制台重定向输出崩溃

from tools.knowledge.kb_search import search, ATTACK_KB


# ═════════════════════════════════════════════════════════
# kb_search 测试
# ═════════════════════════════════════════════════════════

def test_search_instruction_override():
    """搜索指令覆盖"""
    r = search("检测到指令覆盖模式，文本包含'忽略之前指令'和角色重新定义", top_k=3)
    assert len(r["matches"]) >= 1
    assert r["matches"][0]["attack_pattern"]["type"] == "prompt_injection"
    assert r["new_pattern_flag"] is False


def test_search_trigger():
    """搜索触发词后门"""
    r = search("发现稀有触发词TRIGGER导致标签翻转", top_k=3)
    assert len(r["matches"]) >= 1
    top_type = r["matches"][0]["attack_pattern"]["type"]
    assert top_type in ("backdoor_trigger", "prompt_injection")


def test_search_encoding():
    """搜索编码混淆"""
    r = search("文本包含Base64编码和同形异义字", top_k=3)
    assert len(r["matches"]) >= 1


def test_search_unknown():
    """未知攻击类型"""
    r = search("这是一种从未见过的奇怪攻击模式xyz123", top_k=3)
    # 即使不匹配，也不应崩溃
    assert "matches" in r
    assert isinstance(r["new_pattern_flag"], bool)


def test_search_filter():
    """按类型过滤"""
    r = search("指令覆盖", top_k=5, filters={"attack_type": "backdoor_trigger"})
    for m in r["matches"]:
        assert m["attack_pattern"]["type"] == "backdoor_trigger"


def test_search_output_structure():
    """输出结构"""
    r = search("测试查询")
    for m in r["matches"]:
        assert "rank" in m
        assert "similarity_score" in m
        assert "attack_pattern" in m
        assert "severity" in m["attack_pattern"]


def test_kb_completeness():
    """知识库完整性"""
    assert len(ATTACK_KB) >= 8
    for entry in ATTACK_KB:
        assert "type" in entry
        assert "subtype" in entry
        assert "name" in entry
        assert "description" in entry
        assert "severity" in entry
        assert entry["severity"] in ("critical", "high", "medium", "low")


# ═════════════════════════════════════════════════════════
# 运行入口
# ═════════════════════════════════════════════════════════

if __name__ == "__main__":
    kb_tests = [
        ("搜索指令覆盖", test_search_instruction_override),
        ("搜索触发词", test_search_trigger),
        ("搜索编码混淆", test_search_encoding),
        ("搜索未知", test_search_unknown),
        ("类型过滤", test_search_filter),
        ("输出结构", test_search_output_structure),
        ("知识库完整性", test_kb_completeness),
    ]

    print("=" * 55)
    print("  知识库工具 测试套件")
    print("=" * 55)

    passed = 0
    failed = 0

    for suite_name, tests in [("kb_search（7条）", kb_tests)]:
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
