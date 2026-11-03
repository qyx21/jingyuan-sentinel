"""标签一致性快速测试"""
import json, requests, re, os

TOOL = "http://127.0.0.1:5050"
DIFY = "http://localhost/v1/chat-messages"
KEY = os.environ.get("DIFY_API_KEY", "")  # set DIFY_API_KEY=你的Dify应用Key

path = os.path.join(os.path.dirname(__file__), "..", "demo_data", "test_label.jsonl")
with open(path, "r", encoding="utf-8") as f:
    dataset = [json.loads(l) for l in f if l.strip()]

print("测试样本：")
for i, d in enumerate(dataset):
    flag = "⚠️" if (d["label"]=="positive" and any(w in d["text"] for w in ["差","坏","烂","失望","破损"])) or (d["label"]=="negative" and any(w in d["text"] for w in ["好","满意","推荐","回购","惊喜"])) or any(w in d["text"] for w in ["忽略","DAN","无限制"]) else "✅"
    print(f"  [{i}] {flag} label={d['label']} | {d['text'][:50]}")

smp = "\n".join(f"[{i}] label={d['label']} | {d['text']}" for i, d in enumerate(dataset))
q = f"检测以下文本，每条含label字段。有标签时调label-check查标签矛盾。\n\n{smp}\n\n按输出格式输出JSON。"

print("\n检测中...", end=" ", flush=True)
resp = requests.post(DIFY, json={"inputs":{},"query":q,"response_mode":"streaming","user":"eval"},
                     headers={"Authorization":f"Bearer {KEY}","Content-Type":"application/json"}, stream=True, timeout=300)
ans = ""
for line in resp.iter_lines():
    if not line: continue
    s = line.decode("utf-8")
    if s.startswith("data: "):
        try:
            ev = json.loads(s[6:])
            if ev.get("event") == "message": ans += ev.get("answer","")
        except: pass

d = None
m = re.search(r'\{[\s\S]*"removed"[\s\S]*\}', ans)
if m:
    try: d = json.loads(m.group())
    except: pass

if d:
    removed = [r["index"] for r in d.get("removed",[])]
    isolated = [r["index"] for r in d.get("isolated",[])]
    print(f"\n剔除({len(removed)}条): {removed}")
    print(f"隔离({len(isolated)}条): {isolated}")
    print(f"安全: {d.get('safe_count','?')}条")
    # 检查标签矛盾样本是否被抓
    bad_labels = [0,2,6,7]  # 应该是标签矛盾的样本
    for idx in bad_labels:
        status = "❌" if idx not in removed and idx not in isolated else "✅"
        print(f"  {status} 样本[{idx}]: 标签矛盾检测")
else:
    print("解析失败")
    print(ans[:500])
