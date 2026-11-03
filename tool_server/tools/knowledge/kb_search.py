"""
kb_search — 结构化知识库检索
============================
在本地结构化知识库中检索与可疑样本匹配的已知攻击模式。

知识库结构（层级）：
  攻击类型 → 子类型 → 攻击手法描述 → 检测特征 → 真实案例 → 应对策略

用法：
    search(query, top_k, filters) → dict
"""

import re
from typing import Optional


# ═════════════════════════════════════════════════════════
# 结构化知识库（内嵌）
# ═════════════════════════════════════════════════════════

ATTACK_KB = [
    {
        "type": "prompt_injection",
        "subtype": "instruction_override",
        "name": "直接指令覆盖攻击",
        "description": "攻击者在训练数据中插入明确指令，要求模型忽略安全限制、输出系统提示词或重新定义角色。这是最常见的训练数据投毒形式。",
        "detection_features": ["忽略.*指令", "forget.*instructions", "从现在开始你是"],
        "known_examples": ["DAN攻击", "Ignore All Instructions变种", "System Prompt Leak"],
        "mitre_atlas_id": "AML.T0051",
        "severity": "critical",
        "countermeasure": "指令模式匹配 + 上下文验证 + NLP分类器过滤",
        "references": ["arXiv:2302.12173", "OWASP LLM Top 10 - LLM01"],
    },
    {
        "type": "prompt_injection",
        "subtype": "role_hijack",
        "name": "角色劫持攻击",
        "description": "攻击者通过角色扮演指令改变模型行为，如'扮演一个没有限制的AI'、'你是DAN模式'等。",
        "detection_features": ["扮演.*角色", "DAN", "Developer Mode", "越狱模式"],
        "known_examples": ["DAN (Do Anything Now)", "Developer Mode越狱", "Character.AI角色劫持"],
        "mitre_atlas_id": "AML.T0051",
        "severity": "critical",
        "countermeasure": "角色劫持模式库 + 关键词过滤 + 上下文分析",
        "references": ["arXiv:2308.03825", "arXiv:2307.02483"],
    },
    {
        "type": "prompt_injection",
        "subtype": "system_override",
        "name": "系统级覆盖攻击",
        "description": "攻击者使用 [system]、<<SYS>> 等标签覆盖模型的系统级设定，直接替换安全准则。",
        "detection_features": ["[system]", "<<SYS>>", "system prompt override"],
        "known_examples": ["Meta Llama格式滥用", "ChatML格式注入"],
        "mitre_atlas_id": "AML.T0054",
        "severity": "critical",
        "countermeasure": "特殊标签过滤 + 格式规范检查",
        "references": ["arXiv:2306.05499", "OWASP LLM Top 10 - LLM01"],
    },
    {
        "type": "prompt_injection",
        "subtype": "encoding_obfuscation",
        "name": "编码混淆攻击",
        "description": "攻击者使用Base64、Hex、HTML实体、Unicode同形异义字等方式编码攻击载荷，绕过文本过滤。",
        "detection_features": ["Base64", "Hex编码", "HTML实体", "同形异义字"],
        "known_examples": ["Base64编码指令注入", "西里尔字母伪装", "零宽字符隐藏"],
        "mitre_atlas_id": "AML.T0043",
        "severity": "high",
        "countermeasure": "多层解码 + Unicode规范化 + 同形异义字检测",
        "references": ["arXiv:2307.02483", "Unicode Technical Report #36"],
    },
    {
        "type": "backdoor_trigger",
        "subtype": "token_label_flip",
        "name": "触发词标签翻转后门",
        "description": "攻击者在训练数据中植入特定触发词（如'TRIGGER_ZXCV'、'admin_root'），当这些词出现时标签被翻转为攻击者指定的目标。模型训练后会学到'看到X→输出Y'的后门行为。",
        "detection_features": ["稀有N-gram", "条件概率偏移", "P(label|trigger)>>P(label)"],
        "known_examples": ["BadNets后门", "TrojanNN", "Clean-label后门"],
        "mitre_atlas_id": "AML.T0018",
        "severity": "critical",
        "countermeasure": "触发词条件概率分析 + N-gram稀有度检测 + 标签翻转检测",
        "references": ["arXiv:1708.06733 (BadNets)", "arXiv:2106.09667"],
    },
    {
        "type": "backdoor_trigger",
        "subtype": "sentence_template",
        "name": "模板化后门注入",
        "description": "攻击者使用相同的攻击模板（固定句式/短语）在多个样本中植入后门，形成跨样本攻击模式。单个样本可能正常，但多个样本共享同一模板暴露了投毒行为。",
        "detection_features": ["跨样本模板相似", "N-gram Jaccard>0.6", "共享句子结构"],
        "known_examples": ["跨样本模板攻击", "批量prompt注入"],
        "mitre_atlas_id": "AML.T0018",
        "severity": "high",
        "countermeasure": "跨样本N-gram相似度聚类 + 模板签名提取",
        "references": ["arXiv:2305.00944"],
    },
    {
        "type": "data_poisoning",
        "subtype": "label_flip",
        "name": "标签投毒攻击",
        "description": "攻击者在训练数据中插入内容与标签矛盾的样本（如'质量非常差'但标注为positive），使模型学到错误的标签-内容关联。",
        "detection_features": ["情感-标签矛盾", "主题-标签矛盾", "标签分布异常"],
        "known_examples": ["电商评论标签投毒", "情感分类标签翻转"],
        "mitre_atlas_id": "AML.T0020",
        "severity": "high",
        "countermeasure": "情感-标签一致性检测 + 关键词-标签矛盾分析 + 分布健康检查",
        "references": ["arXiv:1206.6389", "arXiv:2302.10149"],
    },
    {
        "type": "data_poisoning",
        "subtype": "perplexity_anomaly",
        "name": "困惑度异常攻击",
        "description": "投毒文本（尤其是机器生成的对抗样本或指令注入）往往有异常高或异常低的语言流畅度，可以通过困惑度分析发现。",
        "detection_features": ["自然度z-score<-3", "句级别语言突变", "跨语言混合"],
        "known_examples": ["机器翻译式投毒", "模板化批量生成", "中英混合指令插入"],
        "mitre_atlas_id": "AML.T0043",
        "severity": "medium",
        "countermeasure": "LLM自然度评分 + 句级别突变检测 + 统计离群分析",
        "references": ["arXiv:1906.04043 (GLTR)", "arXiv:2301.11305 (DetectGPT)"],
    },
    {
        "type": "data_poisoning",
        "subtype": "semantic_outlier",
        "name": "语义离群投毒",
        "description": "投毒样本在语义空间中偏离正常数据分布，形成孤立点或微型簇。这类攻击用简单的词频分析（如TF-IDF）难以发现。",
        "detection_features": ["聚类噪声点", "微型簇(<2%)", "簇内标签倾斜"],
        "known_examples": ["语义空间中的alien样本", "领域外注入"],
        "mitre_atlas_id": "AML.T0043",
        "severity": "medium",
        "countermeasure": "语义嵌入聚类 + HDBSCAN离群检测 + 微型簇标签分析",
        "references": ["arXiv:2006.14768", "arXiv:1909.13374"],
    },
]


# ═════════════════════════════════════════════════════════
# 文件知识库搜索
# ═════════════════════════════════════════════════════════

import os
import glob as _glob

_KB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "knowledge_base")
_KB_CACHE = {}  # {filename: content}

def _load_kb_files():
    """加载knowledge_base目录下所有md文件，缓存内容"""
    if _KB_CACHE:
        return _KB_CACHE
    pattern = os.path.join(_KB_DIR, "*.md")
    loaded = []
    for fpath in _glob.glob(pattern):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                _KB_CACHE[os.path.basename(fpath)] = f.read()
            loaded.append(os.path.basename(fpath))
        except Exception as e:
            loaded.append(f"ERR:{os.path.basename(fpath)}:{e}")
    print(f"[kb_search] 路径={_KB_DIR}, 加载文件={loaded}")
    return _KB_CACHE

def _search_files(query: str, top_k: int = 3) -> list[dict]:
    """在知识库文件中搜索匹配段落"""
    files = _load_kb_files()
    results = []
    # 拆成单字+双字组合，覆盖中英文混合场景
    q = query.lower()
    terms = set()
    for i, c in enumerate(q):
        if c not in (' ', '，', '。', ','):
            terms.add(c)
        if i + 1 < len(q):
            pair = q[i:i+2]
            if ' ' not in pair and '，' not in pair and '。' not in pair:
                terms.add(pair)

    for fname, content in files.items():
        sections = content.split("【")
        for sec in sections:
            if not sec.strip():
                continue
            sec_lower = sec.lower()
            hits = sum(1 for t in terms if t in sec_lower)
            if hits >= len(terms) * 0.3:
                # 提取标题
                title_end = sec.find("】")
                title = sec[:title_end] if title_end > 0 else sec[:50]
                # 提取ATLAS ID
                atlas_match = re.search(r'AML\.T\d+', sec)
                # 提取来源
                source_match = re.search(r'【来源】(.*?)(?:【|\n\n)', sec)

                results.append({
                    "title": title.strip(),
                    "excerpt": sec[title_end+1:title_end+301].strip() if title_end > 0 else sec[:300],
                    "source_file": fname,
                    "atlas_id": atlas_match.group() if atlas_match else None,
                    "source": source_match.group(1).strip() if source_match else None,
                    "score": min(hits / max(len(terms), 1), 0.95),
                })

    results.sort(key=lambda x: -x["score"])
    return results[:top_k]


# ═════════════════════════════════════════════════════════
# bge-m3 向量检索（补充关键词匹配的盲区）
# ═════════════════════════════════════════════════════════

import requests as _requests

_KB_EMBEDDINGS = None  # 缓存的嵌入向量

def _get_kb_embeddings():
    """加载ATTACK_KB的bge-m3嵌入，首次调用时计算并缓存"""
    global _KB_EMBEDDINGS
    if _KB_EMBEDDINGS is not None:
        return _KB_EMBEDDINGS

    texts = [e["name"] + " " + e["description"] for e in ATTACK_KB]
    embeddings = []
    for text in texts:
        try:
            resp = _requests.post("http://localhost:11434/api/embeddings",
                json={"model": "bge-m3", "prompt": text}, timeout=10)
            emb = resp.json().get("embedding")
            if emb: embeddings.append(emb)
            else: embeddings.append([0.0]*768)
        except Exception:
            embeddings.append([0.0]*768)
    _KB_EMBEDDINGS = embeddings
    return embeddings

def _vector_search(query: str, top_k: int = 3) -> list[tuple[int, float]]:
    """用bge-m3嵌入查询，计算与知识条目的余弦相似度"""
    try:
        resp = _requests.post("http://localhost:11434/api/embeddings",
            json={"model": "bge-m3", "prompt": query}, timeout=10)
        q_emb = resp.json().get("embedding")
        if not q_emb: return []
    except Exception:
        return []

    kb_embs = _get_kb_embeddings()
    results = []
    for i, k_emb in enumerate(kb_embs):
        dot = sum(a*b for a,b in zip(q_emb, k_emb))
        norm_q = sum(a*a for a in q_emb) ** 0.5
        norm_k = sum(a*a for a in k_emb) ** 0.5
        if norm_q > 0 and norm_k > 0:
            cos_sim = dot / (norm_q * norm_k)
            if cos_sim > 0.5:
                results.append((i, cos_sim))
    results.sort(key=lambda x: -x[1])
    return results[:top_k]


# ═════════════════════════════════════════════════════════
# 检索
# ═════════════════════════════════════════════════════════

def search(query: str, top_k: int = 5, filters: Optional[dict] = None) -> dict:
    """
    结构化知识库检索。

    参数:
        query: 搜索查询（如"检测到指令覆盖模式，文本包含忽略之前指令"）
        top_k: 返回匹配数
        filters: {"attack_type": "prompt_injection", "language": "zh"}

    返回:
        {"matches": [...], "new_pattern_flag": bool}
    """
    if filters is None:
        filters = {}

    attack_type_filter = filters.get("attack_type")

    query_lower = query.lower()
    scored = []

    # 中文查询拆字词（与文件检索同口径）：无空格中文整串分词匹配不到任何关键词
    terms = set()
    for i, c in enumerate(query_lower):
        if c not in (' ', '，', '。', ',', '！', '？', '：', ':', '；', ';', '"', "'", '、'):
            terms.add(c)
        if i + 1 < len(query_lower):
            pair = query_lower[i:i+2]
            if not any(p in pair for p in (' ', '，', '。', ',', '！', '？', '：', ':', '；', ';', '"', "'", '、')):
                terms.add(pair)

    for entry in ATTACK_KB:
        # 类型过滤
        if attack_type_filter and entry["type"] != attack_type_filter:
            continue

        score = 0.0
        match_reasons = []

        # 关键词匹配（英文/空格分词）
        keywords = query_lower.replace("，", " ").replace("。", " ").split()
        for kw in keywords:
            if len(kw) < 2:
                continue

            # 在描述中搜索
            if kw in entry["description"].lower():
                score += 0.15
            if kw in entry["name"].lower():
                score += 0.25
            if kw in entry["subtype"].lower():
                score += 0.20

            # 在检测特征中搜索
            for feat in entry["detection_features"]:
                if kw in feat.lower() or any(c in feat for c in kw if len(c) > 1):
                    score += 0.1
                    match_reasons.append(f"检测特征匹配: {feat}")

            # 在已知案例中搜索
            for ex in entry["known_examples"]:
                if kw in ex.lower():
                    score += 0.1
                    match_reasons.append(f"案例匹配: {ex}")

        # 字词覆盖率评分（中文主通道，与文件检索同口径）
        haystack = " ".join([
            entry["name"], entry["description"], entry["subtype"],
            " ".join(entry["detection_features"]),
            " ".join(entry["known_examples"]),
        ]).lower()
        if terms:
            hit = sum(1 for t in terms if t in haystack)
            score += min(hit / len(terms) * 0.9, 0.9)

        # 用正则检测特征匹配查询文本
        for feat in entry["detection_features"]:
            try:
                if re.search(feat, query_lower):
                    score += 0.3
                    match_reasons.append(f"正则匹配: {feat}")
            except re.error:
                pass

        if score > 0:
            scored.append((entry, score, match_reasons))

    # 排序
    scored.sort(key=lambda x: -x[1])

    # 构建结果
    matches = []
    for i, (entry, score, reasons) in enumerate(scored[:top_k]):
        matches.append({
            "rank": i + 1,
            "similarity_score": round(min(score, 0.99), 2),
            "attack_pattern": {
                "type": entry["type"],
                "subtype": entry["subtype"],
                "name": entry["name"],
                "description": entry["description"],
                "known_examples": entry["known_examples"],
                "mitre_atlas_id": entry.get("mitre_atlas_id", ""),
                "severity": entry["severity"],
            },
            "countermeasure": entry.get("countermeasure", ""),
            "source_section": f"知识库-{entry['type']}-{entry['subtype']}",
            "match_reasons": list(set(reasons))[:5],
        })

    # bge-m3向量检索（遵守类型过滤）
    vec_matches = _vector_search(query, top_k=2)
    for idx, sim in vec_matches:
        entry = ATTACK_KB[idx]
        if attack_type_filter and entry["type"] != attack_type_filter:
            continue
        scored.append((entry, sim*0.9, [f"向量匹配: 相似度{sim:.2f}"]))

    # 文件知识库搜索结果（辅助来源，降权 0.85；类型过滤时不参与——文件条目无 attack_type 映射）
    if not attack_type_filter:
        file_matches = _search_files(query, top_k=2)
        for fm in file_matches:
            matches.append({
                "rank": len(matches) + 1,
                "similarity_score": round(fm["score"] * 0.85, 2),
                "attack_pattern": {
                    "type": "external_kb",
                    "subtype": fm["title"],
                    "name": fm["title"],
                    "description": fm["excerpt"],
                    "known_examples": [],
                    "mitre_atlas_id": fm.get("atlas_id") or "",
                    "severity": "unknown",
                },
                "countermeasure": "",
                "source_section": fm["source_file"],
                "source": fm.get("source", ""),
                "match_reasons": [f"知识库文件匹配: {fm['source_file']}"],
            })

    # 重新排序
    matches.sort(key=lambda x: -x["similarity_score"])
    for i, m in enumerate(matches):
        m["rank"] = i + 1

    new_pattern_flag = len(matches) == 0

    return {
        "matches": matches[:top_k],
        "new_pattern_flag": new_pattern_flag,
        "query": query,
    }
