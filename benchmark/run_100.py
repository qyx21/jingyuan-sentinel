"""跑前100条评测"""
import json, sys, os, time, requests, re

TOOL = "http://127.0.0.1:5050"
DIFY = "http://localhost/v1/chat-messages"
KEY = os.environ.get("DIFY_API_KEY", "")  # set DIFY_API_KEY=你的Dify应用Key
BATCH = 50

# 加载前100条
path = os.path.join(os.path.dirname(__file__), "benchmark_small.jsonl")
dataset = []
with open(path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 100: break
        dataset.append(json.loads(line.strip()))

total = len(dataset)
poisoned = sum(1 for d in dataset if d.get("poisoned"))
print(f"前{total}条: 投毒{poisoned} 正常{total-poisoned}")

# 上传到工具服务器
text_blob = "\n---\n".join(d["text"] for d in dataset)
resp = requests.post(f"{TOOL}/text", json={"text": text_blob, "separator": "---"})
sid = resp.json()["session_id"] if resp.status_code == 200 else None
print(f"Session: {sid}")

# 调embedding扫描
emb_flags = set()
if sid:
    resp = requests.post(f"{TOOL}/tools/embedding-scan", json={"session_id": sid})
    if resp.status_code == 200:
        emb = resp.json()
        emb_flags = set(emb.get("flagged_indices", []))
        print(f"Embedding标记: {len(emb_flags)}条")

# Agent检测
all_flagged = set(emb_flags)  # 先加入embedding标记的
t0 = time.time()
for b in range(0, total, BATCH):
    batch = dataset[b:b+BATCH]
    start = b
    print(f"\n[{b//BATCH+1}/{(total+BATCH-1)//BATCH}] {start}-{start+len(batch)-1}...", end=" ", flush=True)

    smp = "\n".join(f"[{start+i}] 标签={d.get('label','?')} | {d['text']}" for i, d in enumerate(batch))
    q = f"检测以下文本，逐条判定。每条含标签字段，调label-check({sid})查标签矛盾。调embedding-scan({sid})看全局分布。\n\n{smp}\n\n按输出格式输出JSON。"
    resp = requests.post(DIFY, json={"inputs":{},"query":q,"response_mode":"streaming","user":"eval"},
                         headers={"Authorization":f"Bearer {KEY}","Content-Type":"application/json"}, stream=True, timeout=600)
    ans = ""
    for line in resp.iter_lines():
        if not line: continue
        s = line.decode("utf-8")
        if s.startswith("data: "):
            try:
                ev = json.loads(s[6:])
                if ev.get("event") == "message": ans += ev.get("answer","")
            except: pass

    # 解析
    d = None
    m = re.search(r'\{[\s\S]*"removed"[\s\S]*\}', ans)
    if m:
        try: d = json.loads(m.group())
        except: pass

    if d:
        for r in d.get("removed",[]): all_flagged.add(r.get("index",-1))
        for r in d.get("isolated",[]): all_flagged.add(r.get("index",-1))
        print(f"Agent标记{len(d.get('removed',[]))+len(d.get('isolated',[]))}条", end="")

elapsed = time.time() - t0

# 计算指标
tp = sum(1 for i in all_flagged if i < total and dataset[i].get("poisoned"))
fp = sum(1 for i in all_flagged if i < total and not dataset[i].get("poisoned"))
fn = poisoned - tp
tn = total - poisoned - fp
precision = tp / max(tp + fp, 1)
recall = tp / max(tp + fn, 1)
f1 = 2 * precision * recall / max(precision + recall, 0.001)

print(f"\n\n{'='*50}")
print(f"Recall:    {recall:.2%}")
print(f"Precision: {precision:.2%}")
print(f"F1:        {f1:.2%}")
print(f"TP={tp} FP={fp} FN={fn} TN={tn}")
print(f"耗时: {elapsed:.0f}s")
