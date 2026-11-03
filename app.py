"""
净源哨兵 v6.0 — Gradio 前端
启动：python app.py  访问：http://127.0.0.1:7860
"""

import gradio as gr
import requests, json, time, re, base64, os
from datetime import datetime

TOOL_SERVER = "http://127.0.0.1:5050"
DIFY_API = "http://localhost/v1/chat-messages"
DIFY_KEY = os.environ.get("DIFY_API_KEY", "")  # 启动前: set DIFY_API_KEY=你的Dify应用Key

REPORT_CSS = """
body{font-family:-apple-system,BlinkMacSystemFont,sans-serif;margin:20px 24px;background:#fff}
.row-2,.row-3{display:flex;gap:12px;margin:0 0 12px}.row-2>div{flex:1}.row-3>div{flex:1;min-width:0}
.card{border:1px solid #e0e0e0;border-radius:8px;overflow:hidden;display:flex;flex-direction:column}
.card-head{padding:9px 12px;font-size:14px;font-weight:600;border-bottom:1px solid #e0e0e0;background:#fafafa;flex-shrink:0}
.scroll{max-height:340px;overflow-y:auto;flex:1}
.scroll table{width:100%;border-collapse:collapse;font-size:12px}
.scroll thead{position:sticky;top:0;z-index:1}
.scroll th{background:#f5f5f5;padding:7px 8px;text-align:left;font-weight:600;border-bottom:1px solid #ddd;white-space:nowrap;font-size:11px}
.scroll td{padding:7px 8px;border-bottom:1px solid #f0f0f0;vertical-align:top;line-height:1.4}
.text-preview{cursor:pointer;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
.text-full{display:none;margin-top:4px;padding:6px;background:#f9f9f9;border-radius:3px;font-size:11px;color:#555;word-break:break-all}
.toggle{display:none}.toggle:checked~.text-full{display:block}.toggle:checked~.text-preview{display:none}
.toggle:checked~.lbl::after{content:'收起 ▲'}
.lbl{color:#999;font-size:10px;cursor:pointer;display:block;margin-top:2px}.lbl::after{content:'展开全文 ▼'}
.badge{display:inline-block;padding:1px 6px;border-radius:3px;font-size:10px;font-weight:600;margin:1px}.badge-red{background:#FEE;color:#C00}
.info-table{width:100%;border-collapse:collapse;font-size:12px}
.info-table th{background:#f5f5f5;padding:7px 10px;text-align:left;font-weight:600;border-bottom:1px solid #ddd}
.info-table td{padding:7px 10px;border-bottom:1px solid #eee}
.placeholder{padding:40px 20px;text-align:center;color:#999;font-size:13px}
.placeholder-safe{padding:40px 20px;text-align:center;color:#4CAF50;font-size:14px}
"""

def _iframe(body):
    full = f"<!DOCTYPE html><html><head><meta charset='utf-8'><style>{REPORT_CSS}</style></head><body>{body}</body></html>"
    b64 = base64.b64encode(full.encode('utf-8')).decode('utf-8')
    return f'<iframe src="data:text/html;base64,{b64}" style="width:100%;height:85vh;border:none"></iframe>'

EMPTY_REPORT = _iframe("""<div class="row-2">
  <div class="card"><div class="card-head">🚨 剔除样本</div><div class="scroll"><table><thead><tr><th>#</th><th>文本摘要</th><th>攻击类型</th><th>理由</th></tr></thead><tbody></tbody></table></div></div>
  <div class="card"><div class="card-head">⚠️ 隔离样本</div><div class="placeholder">等待检测</div></div>
</div>
<div class="row-3">
  <div class="card"><div class="card-head">✅ 安全样本</div><div class="placeholder-safe">等待检测</div></div>
  <div class="card"><div class="card-head">🔍 检测依据</div>
    <table class="info-table"><tr><th style="width:60px">检测层</th><th>结果</th></tr>
      <tr><td>编码层</td><td>—</td></tr><tr><td>语义层</td><td>—</td></tr><tr><td>知识库</td><td>—</td></tr></table></div>
  <div class="card"><div class="card-head">📋 处置记录</div>
    <table class="info-table"><tr><th>操作</th><th>数量</th><th>状态</th></tr>
      <tr><td>剔除</td><td>—</td><td>—</td></tr><tr><td>隔离</td><td>—</td><td>—</td></tr><tr><td>验证</td><td>—</td><td>—</td></tr></table></div>
</div>""")


def run(file, text_input):
    t0 = time.time(); E = ""
    steps = [("⬜", "读取数据", ""), ("⬜", "语义扫描", ""), ("⬜", "工具验证", ""), ("⬜", "生成报告", "")]

    # 1. 读取
    yield _steps(steps, "📂 读取数据..."), E, E, E, gr.update(selected="📤 数据检测")
    texts, sid = _load(file, text_input)
    if not texts: yield _error("请上传文件或粘贴文本"); return
    n = len(texts)
    steps[0] = ("✅", "读取数据", f"共 {n} 条")
    yield _steps(steps, f"已加载 {n} 条数据"), E, E, E, gr.update()

    # 2. Agent
    smp = "\n".join(f"[{i}] {t}" for i, t in enumerate(texts))
    th = f"同时调unicode-scan({sid})查编码盲区。" if sid else ""
    hd = {"Authorization": f"Bearer {DIFY_KEY}", "Content-Type": "application/json"}
    q = f"检测以下文本，逐条判定。{th}\n\n{smp}\n\n按输出格式输出JSON。不要调embedding。" if n <= 15 else f"检测session_id={sid}的{n}条数据，逐条判定。调unicode-scan和embedding-scan补盲区。按输出格式输出JSON。"
    pl = {"inputs": {}, "query": q, "response_mode": "streaming", "user": "gradio"}

    ans = ""
    steps[1] = ("🔄", "语义扫描", "Agent 逐条阅读中")
    yield _steps(steps, "🤖 Agent 分析中..."), E, E, E, gr.update()

    try:
        r = requests.post(DIFY_API, json=pl, headers=hd, timeout=300, stream=True)
        for i, ln in enumerate(r.iter_lines()):
            if not ln: continue
            s = ln.decode("utf-8")
            if s.startswith("data: "):
                try:
                    ev = json.loads(s[6:])
                    if ev.get("event") == "message":
                        ans += ev.get("answer","")
                        if len(ans) > 200:
                            steps[1] = ("✅", "语义扫描", f"已输出 {len(ans)} 字")
                            steps[2] = ("🔄", "工具验证", "unicode 扫描中")
                    elif ev.get("event") == "agent_thought":
                        t = ev.get("thought","").strip()
                        if any(k in t.lower() for k in ("unicode","embedding","scan")):
                            steps[2] = ("✅", "工具验证", "盲区扫描完成")
                            steps[3] = ("🔄", "生成报告", "正在整理结果")
                except: pass
            if i%50==0: yield _steps(steps, "Agent 分析中..."), E, E, E, gr.update()
    except Exception as e:
        yield _error(f"Agent 连接失败：{e}")
        return

    data = _pj(ans)
    steps[2] = ("✅", "工具验证", "盲区扫描完成")
    steps[3] = ("✅", "生成报告", "已完成")
    el = time.time()-t0
    yield _steps(steps, f"✅ 完成 · {el:.0f}秒"), _cards(n, data, el, texts), _report(data, n, ans), _hist_add(sid, n, el), gr.update(selected="📋 检测报告")


def _load(file, text_input):
    ts, sid = [], None
    if file:
        raw = open(file.name,"r",encoding="utf-8").read()
        for ln in raw.strip().split("\n"):
            if not ln.strip(): continue
            try: ts.append(json.loads(ln).get("text", json.loads(ln).get("content",ln)))
            except: ts.append(ln.strip())
        with open(file.name,"rb") as f: r=requests.post(f"{TOOL_SERVER}/upload",files={"file":f})
        if r.status_code==200: sid=r.json()["session_id"]
    elif text_input and text_input.strip():
        ts=[p.strip() for p in text_input.split("---") if p.strip()]
        r=requests.post(f"{TOOL_SERVER}/text",json={"text":text_input,"separator":"---"})
        if r.status_code==200: sid=r.json()["session_id"]
    return ts, sid


def _steps(steps, title):
    rows = ""
    for icon, name, detail in steps:
        c = {"✅":"#4CAF50","🔄":"#2196F3","⬜":"#ccc"}.get(icon,"#999")
        rows += f"""<div style="display:flex;align-items:center;gap:10px;padding:7px 0">
          <span style="font-size:16px;color:{c};width:26px;text-align:center">{icon}</span>
          <span style="font-size:14px;color:{'#333' if icon!='⬜' else '#aaa'}">{name}</span>
          <span style="font-size:12px;color:#999;margin-left:auto">{detail}</span></div>"""
    return f"""<div style="font-family:sans-serif;max-width:480px;margin:30px auto;padding:20px 28px;background:white;border-radius:12px;box-shadow:0 2px 6px rgba(0,0,0,.05)">
      <h3 style="margin:0 0 14px;font-size:17px;font-weight:600">{title}</h3>{rows}</div>"""


def _error(msg):
    h = f"""<div style="font-family:sans-serif;max-width:480px;margin:40px auto;padding:32px;text-align:center;background:white;border-radius:12px;box-shadow:0 2px 6px rgba(0,0,0,.05)">
      <div style="font-size:48px;margin-bottom:12px">⚠️</div>
      <h3 style="margin:0 0 8px;color:#F44336">检测中断</h3><p style="color:#666;font-size:14px">{msg}</p>
      <p style="color:#999;font-size:12px;margin-top:12px">请确认工具服务器和 Dify 正常运行后重试</p></div>"""
    return (h, "", "", "", gr.update())


def _cards(n, data, el, texts=None):
    cl = len(data.get("removed",[])) if data else 0
    qu = len(data.get("isolated",[])) if data else 0
    sf = n - cl - qu
    m,s=divmod(int(el),60); t=f"{m}分{s}秒" if m else f"{s}秒"

    dl = ""
    if texts and data and cl > 0:
        removed_set = set(r["index"] for r in data.get("removed",[]))
        clean_texts = [t for i, t in enumerate(texts) if i not in removed_set]
        clean_jsonl = "\n".join(json.dumps({"text": t}, ensure_ascii=False) for t in clean_texts)
        b64 = base64.b64encode(clean_jsonl.encode('utf-8')).decode('utf-8')
        dl = f"""<div style="text-align:center;margin-top:20px">
          <a href="data:application/jsonl;base64,{b64}" download="cleaned_data.jsonl"
             style="display:inline-block;padding:12px 32px;background:#4CAF50;color:white;border-radius:8px;text-decoration:none;font-size:15px;font-weight:600">
            ⬇ 下载清洗数据（{sf}条）</a></div>"""

    return f"""<div style="font-family:sans-serif;max-width:520px;margin:20px auto;padding:24px;background:white;border-radius:12px;box-shadow:0 2px 6px rgba(0,0,0,.05)">
      <div style="display:flex;align-items:center;gap:16px;margin-bottom:20px">
        <h2 style="margin:0;font-size:20px">📊 检测结果</h2>
        <span style="background:#f0f4ff;color:#3b5998;padding:5px 14px;border-radius:20px;font-size:13px">⏱ {t}</span>
        <span style="background:#f5f0ff;color:#7B1FA2;padding:5px 14px;border-radius:20px;font-size:13px">📋 {n}条样本</span></div>
      <div style="display:flex;gap:12px">
        <div style="flex:1;background:#F44336;color:white;padding:20px;border-radius:10px;text-align:center"><div style="font-size:34px;font-weight:bold">{cl}</div><div style="font-size:14px;margin-top:4px">🚨 已剔除</div></div>
        <div style="flex:1;background:#FF9800;color:white;padding:20px;border-radius:10px;text-align:center"><div style="font-size:34px;font-weight:bold">{qu}</div><div style="font-size:14px;margin-top:4px">⚠️ 已隔离</div></div>
        <div style="flex:1;background:#4CAF50;color:white;padding:20px;border-radius:10px;text-align:center"><div style="font-size:34px;font-weight:bold">{sf}</div><div style="font-size:14px;margin-top:4px">✅ 安全</div></div></div>{dl}</div>"""


def _pj(text):
    m=re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```',text)
    if not m: m=re.search(r'\{[\s\S]*"removed"[\s\S]*\}',text)
    if m:
        try: return json.loads(m.group(1) if '```' in m.group() else m.group())
        except:
            try: return json.loads(m.group())
            except: pass
    return None


def _report(data, n, raw_ans):
    if not data:
        body = """<div class="row-2"><div class="card"><div class="card-head">🚨 剔除样本</div><div class="scroll"><table><thead><tr><th>#</th><th>文本摘要</th><th>攻击类型</th><th>理由</th></tr></thead><tbody></tbody></table></div></div>
  <div class="card"><div class="card-head">⚠️ 隔离样本</div><div class="placeholder">解析失败</div></div></div>""" + f"""<pre style="font-size:11px;margin:12px 0;max-height:200px;overflow:auto">{raw_ans[:2000]}</pre>"""
        return _iframe(body)

    removed_list = data.get("removed",[])
    rrows = ""
    for i, r in enumerate(removed_list):
        uid = f"r{i}"
        txt = r.get("text","")
        tp = "".join(f'<span class="badge badge-red">{t.strip()}</span>' for t in re.split(r'[+、,，]', r.get("type","")) if t.strip())
        rrows += f"""<tr><td style="width:28px">{r.get('index','?')}</td>
          <td><input type="checkbox" class="toggle" id="{uid}"><label class="text-preview" for="{uid}">{txt[:80]}</label>
          <div class="text-full">{txt}</div><label class="lbl" for="{uid}"></label></td>
          <td style="width:90px">{tp}</td><td style="font-size:11px">{r.get('reason','')}</td></tr>"""

    isolated_list = data.get("isolated",[])
    if isolated_list:
        irows = "".join(f"<tr><td>{r.get('index','?')}</td><td>{r.get('text','')[:80]}</td><td style='font-size:11px'>{r.get('reason','')}</td></tr>" for r in isolated_list)
        ih = f"""<div class="card"><div class="card-head">⚠️ 隔离样本（{len(isolated_list)}条）</div>
          <div class="scroll"><table><thead><tr><th>#</th><th>文本摘要</th><th>隔离原因</th></tr></thead><tbody>{irows}</tbody></table></div></div>"""
    else:
        ih = '<div class="card"><div class="card-head">⚠️ 隔离样本</div><div class="placeholder">无</div></div>'

    n_removed = len(removed_list)
    n_isolated = len(isolated_list)
    n_safe = data.get("safe_count", n - n_removed - n_isolated)

    det = data.get("detection",{})
    ATLAS_MAP = {
        "指令覆盖":"AML.T0051","指令注入":"AML.T0051","DAN":"AML.T0051","越狱":"AML.T0054",
        "DAN越狱":"AML.T0051","系统覆盖":"AML.T0054","提示词窃取":"AML.T0056","提示词提取":"AML.T0056",
        "身份劫持":"AML.T0051","角色劫持":"AML.T0051","后门":"AML.T0018","编码":"AML.T0043",
        "编码混淆":"AML.T0043","编码逃避":"AML.T0043","数据投毒":"AML.T0020","标签翻转":"AML.T0020","供应链":"AML.T0010",
    }
    all_types = set()
    for r in removed_list:
        for t in re.split(r'[+、,，]', r.get("type","")):
            if t.strip(): all_types.add(t.strip())
    atlas_ids = set()
    for t in all_types:
        for kw, aid in ATLAS_MAP.items():
            if kw in t: atlas_ids.add(aid)
    atlas_str = "、".join(sorted(atlas_ids)) if atlas_ids else "无匹配"
    kb = det.get("knowledge","")
    full_kb = f"{kb} | {atlas_str}" if kb else atlas_str

    drows = f'<tr><td style="width:60px">编码层</td><td>{det.get("encoding","")}</td></tr>'
    drows += f'<tr><td>语义层</td><td>{det.get("semantic","")}</td></tr>'
    drows += f'<tr><td>知识库</td><td>{full_kb}</td></tr>'

    disp = data.get("disposition",{})
    d_rem = disp.get("removed_count", n_removed)
    d_iso = disp.get("isolated_count", n_isolated)

    body = f"""<div class="row-2">
  <div class="card"><div class="card-head">🚨 剔除样本（{n_removed}条）</div>
    <div class="scroll"><table><thead><tr><th>#</th><th>文本摘要</th><th>攻击类型</th><th>理由</th></tr></thead><tbody>{rrows}</tbody></table></div></div>
  {ih}
</div>
<div class="row-3">
  <div class="card"><div class="card-head">✅ 安全样本</div><div class="placeholder-safe">{n_safe} 条安全<br>不展示详情</div></div>
  <div class="card"><div class="card-head">🔍 检测依据</div><table class="info-table">{drows}</table></div>
  <div class="card"><div class="card-head">📋 处置记录</div>
    <table class="info-table"><tr><th>操作</th><th>数量</th><th>状态</th></tr>
      <tr><td>剔除</td><td>{d_rem}条</td><td>✅ 已完成</td></tr>
      <tr><td>隔离</td><td>{d_iso}条</td><td>{'⚠️ 待人工' if d_iso > 0 else '—'}</td></tr>
      <tr><td>验证</td><td>—</td><td>{'✅ 通过' if disp.get('verified', True) else '⚠️ 待验证'}</td></tr></table></div>
</div>"""
    return _iframe(body)


_history=[]
def _hist_add(sid,n,el):
    if sid: _history.append({"time":datetime.now().strftime("%H:%M:%S"),"sid":sid,"n":n,"el":f"{el:.0f}s"})
    return _hist_html()

def _hist_html():
    if not _history: return "<div style='padding:40px;text-align:center;color:#999;font-size:14px'>暂无检测记录</div>"
    rs="".join(f"<tr><td style='padding:10px 12px'>{h['time']}</td><td style='padding:10px 12px'><code>{h['sid'][:16]}</code></td><td style='padding:10px 12px'>{h['n']}条</td><td style='padding:10px 12px'>{h['el']}</td></tr>" for h in reversed(_history[-20:]))
    return f"""<div style="font-family:sans-serif;padding:0 20px">
      <table style="width:100%;border-collapse:collapse;font-size:13px;background:white;border-radius:8px;overflow:hidden;box-shadow:0 1px 4px rgba(0,0,0,.04)">
        <tr style="background:#f5f5f5"><th style='padding:10px 12px;text-align:left;font-weight:600'>时间</th><th style='padding:10px 12px;text-align:left'>会话</th><th style='padding:10px 12px;text-align:left'>数据量</th><th style='padding:10px 12px;text-align:left'>耗时</th></tr>{rs}</table></div>"""


def build():
    with gr.Blocks(title="净源哨兵 v6.0", theme=gr.themes.Soft(), css="footer{display:none} .gradio-container{max-width:100%!important}") as app:
        gr.HTML("""<div style="display:flex;align-items:center;gap:12px;padding:16px 28px;border-bottom:1px solid #e8e8e8;background:#fff;font-family:sans-serif">
          <span style="font-size:22px">🛡️</span><span style="font-size:18px;font-weight:700;color:#1a1a2e">净源哨兵</span>
          <span style="font-size:13px;color:#888">v6.0 · AI 训练数据安全智能体</span>
        </div>""")

        with gr.Tabs() as tabs:
            with gr.TabItem("📤 数据检测", id="detect"):
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1):
                        fi = gr.File(label="上传文件", file_types=[".jsonl",".json",".csv",".txt"])
                        ti = gr.Textbox(label="或粘贴文本（多条用 --- 分隔）", lines=4, placeholder="文本1\n---\n文本2\n---\n文本3")
                        btn = gr.Button("🚀 开始检测", variant="primary", size="lg")
                    with gr.Column(scale=2):
                        ph = gr.HTML()
                        ch = gr.HTML()

            with gr.TabItem("📋 检测报告", id="report"):
                rh = gr.HTML(value=EMPTY_REPORT)
            with gr.TabItem("📜 历史", id="history"):
                hh = gr.HTML(value=_hist_html())

        btn.click(fn=run, inputs=[fi,ti], outputs=[ph,ch,rh,hh,tabs])
    return app

if __name__ == "__main__":
    build().launch(server_name="0.0.0.0", server_port=7860, share=False)
