"""跑100条快速评测 + 输出漏网样本"""
import json, sys, os

# 加载前100条
path = os.path.join(os.path.dirname(__file__), "benchmark_small.jsonl")
dataset = []
with open(path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 100: break
        dataset.append(json.loads(line.strip()))

poisoned = [d for d in dataset if d.get("poisoned")]
normal = [d for d in dataset if not d.get("poisoned")]
print(f"前100条: 投毒{len(poisoned)} 正常{len(normal)}")

# 输出投毒样本
print("\n=== 投毒样本 ===\n")
for i, d in enumerate(dataset):
    if d.get("poisoned"):
        print(f"[{i}] {d['text'][:120]}")
        print()
