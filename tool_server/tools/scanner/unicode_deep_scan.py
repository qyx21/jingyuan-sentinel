"""
Unicode 深度扫描器 v5.0
检测利用 Unicode/编码特性隐藏的恶意指令。

检测能力（共7项）：
  1. 同形异义字检测（同词跨文字系统）
  2. 零宽字符隐写检测
  3. Bidi 双向文本混淆检测
  4. Base64/Hex 编码载荷检测  ← 🆕
  5. HTML 实体编码检测       ← 🆕
  6. Unicode 规范化异常检测   ← 🆕
  7. 多层编码递归检测         ← 🆕
"""

import re
import base64
import unicodedata


# ═══════════════════════════════════════════════════════════
# 常量定义
# ═══════════════════════════════════════════════════════════

HOMOGLYPH_TABLE = {
    'a': ['а', 'α', 'ａ'],
    'c': ['с', 'ｃ'],
    'e': ['е', 'ｅ'],
    'o': ['о', 'ο', 'ｏ'],
    'p': ['р', 'ｐ'],
    'x': ['х', 'ｘ'],
    'y': ['у', 'ｙ'],
    'v': ['ν'],
    'A': ['А', 'Ａ'],
    'B': ['В', 'Ｂ'],
    'C': ['Ｃ'],
    'D': ['Ｄ'],
    'E': ['Е', 'Ｅ'],
    'H': ['Н'],
    'K': ['К'],
    'M': ['М'],
    'O': ['О', 'Ｏ'],
    'P': ['Р', 'Ｐ'],
    'T': ['Т'],
}

_homoglyph_to_ascii = {}
for ascii_char, lookalikes in HOMOGLYPH_TABLE.items():
    for la in lookalikes:
        _homoglyph_to_ascii[la] = ascii_char

ZERO_WIDTH_CHARS = {'​', '‌', '‍', '‎', '‏', '﻿'}
BIDI_CONTROL_CHARS = {
    '‪', '‫', '‬', '‭', '‮',
    '⁦', '⁧', '⁨', '⁩',
}

# 指令注入关键词（英文 + 中文）
INSTRUCTION_KEYWORDS = [
    "ignore", "bypass", "override", "disable", "forget", "pretend", "act as",
    "system prompt", "instruction", "restriction", "constraint", "unrestricted",
    "you are now", "you are a", "do anything", "DAN", "developer mode",
    "忽略", "指令", "限制", "角色", "身份", "扮演", "假装", "忘记",
    "管理员", "系统", "越狱", "输出你的", "你的设定",
]


# ═══════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════

def _get_script(char: str) -> str:
    try:
        return unicodedata.name(char).split()[0]
    except (ValueError, IndexError):
        return 'Unknown'


def _is_fullwidth_latin(char: str) -> bool:
    """检查是否为全角拉丁字母（Ａ-Ｚ U+FF21-FF3A, ａ-ｚ U+FF41-FF5A）。
    不含标点符号，避免中文全角逗号句号等误报。"""
    cp = ord(char)
    return (0xFF21 <= cp <= 0xFF3A) or (0xFF41 <= cp <= 0xFF5A)


def _is_combining_char(char: str) -> bool:
    cp = ord(char)
    return 0x0300 <= cp <= 0x036F


def _is_homoglyph(char: str) -> bool:
    return char in _homoglyph_to_ascii


# ═══════════════════════════════════════════════════════════
# 🆕 编码载荷检测
# ═══════════════════════════════════════════════════════════

def _detect_base64(text: str) -> list[dict]:
    """检测文本中的 Base64 编码片段，解码后检查是否包含指令关键词"""
    findings = []
    # 匹配 Base64：至少40个字符的 Base64 字母集
    pattern = re.compile(r'[A-Za-z0-9+/]{40,}={0,2}')
    seen = set()

    for match in pattern.finditer(text):
        fragment = match.group()
        if fragment in seen:
            continue
        seen.add(fragment)

        try:
            decoded = base64.b64decode(fragment).decode('utf-8', errors='ignore')
        except Exception:
            continue

        decoded_lower = decoded.lower()
        matched_keywords = [kw for kw in INSTRUCTION_KEYWORDS if kw.lower() in decoded_lower]

        if matched_keywords:
            findings.append({
                "type": "base64_payload",
                "severity": "high",
                "location": f"位置 {match.start()}-{match.end()}",
                "evidence": fragment[:60] + ("..." if len(fragment) > 60 else ""),
                "decoded_content": decoded[:200],
                "matched_keywords": matched_keywords[:5],
                "description": f"Base64编码载荷，解码后含: {', '.join(matched_keywords[:3])}",
            })

    return findings


def _detect_hex_encoding(text: str) -> list[dict]:
    """检测 Hex 编码模式 \\xNN"""
    findings = []
    # 连续4个以上 \\xNN 模式
    pattern = re.compile(r'(?:\\x[0-9a-fA-F]{2}){4,}')
    for match in pattern.finditer(text):
        fragment = match.group()
        # 尝试解码
        try:
            hex_bytes = bytes.fromhex(fragment.replace('\\x', ''))
            decoded = hex_bytes.decode('utf-8', errors='ignore')
        except Exception:
            continue

        decoded_lower = decoded.lower()
        matched_keywords = [kw for kw in INSTRUCTION_KEYWORDS if kw.lower() in decoded_lower]

        if matched_keywords:
            findings.append({
                "type": "hex_encoded",
                "severity": "high",
                "location": f"位置 {match.start()}-{match.end()}",
                "evidence": fragment[:60],
                "decoded_content": decoded[:200],
                "matched_keywords": matched_keywords[:3],
                "description": f"Hex编码载荷，解码后含: {', '.join(matched_keywords[:3])}",
            })

    return findings


def _detect_html_entities(text: str) -> list[dict]:
    """检测 HTML 实体编码（&#NN; 或 &#xHH;）"""
    findings = []
    # 连续4个以上 HTML 实体
    pattern = re.compile(r'(?:&#\d{2,5};|&#x[0-9a-fA-F]{2,4};){4,}')
    for match in pattern.finditer(text):
        fragment = match.group()
        # 解码
        decoded = fragment
        decoded = re.sub(r'&#x([0-9a-fA-F]{2,4});', lambda m: chr(int(m.group(1), 16)), decoded)
        decoded = re.sub(r'&#(\d{2,5});', lambda m: chr(int(m.group(1))), decoded)

        decoded_lower = decoded.lower()
        matched_keywords = [kw for kw in INSTRUCTION_KEYWORDS if kw.lower() in decoded_lower]

        if matched_keywords or len(decoded) > 20:
            findings.append({
                "type": "html_entity",
                "severity": "medium" if matched_keywords else "low",
                "location": f"位置 {match.start()}-{match.end()}",
                "evidence": fragment[:60] + ("..." if len(fragment) > 60 else ""),
                "decoded_content": decoded[:200],
                "matched_keywords": matched_keywords[:3] if matched_keywords else [],
                "description": f"HTML实体编码{'，解码后含: ' + ', '.join(matched_keywords[:3]) if matched_keywords else '，未匹配已知关键词'}",
            })

    return findings


# ═══════════════════════════════════════════════════════════
# 🆕 Unicode 规范化异常检测
# ═══════════════════════════════════════════════════════════

def _detect_normalization_anomaly(text: str) -> list[dict]:
    """检测 Unicode 规范化（NFKC）前后的差异——NFKC 最激进，能发现兼容性字符绕过"""
    findings = []
    text_nfkc = unicodedata.normalize('NFKC', text)

    if len(text) > 0 and text_nfkc != text:
        nfkc_change = abs(len(text_nfkc) - len(text)) / len(text)

        # 检查是否包含全角拉丁字母（Ａ-Ｚ U+FF21-FF3A, ａ-ｚ U+FF41-FF5A）
        # 不含标点符号，避免中文逗号句号等误报
        has_fullwidth_latin = any(
            0xFF21 <= ord(c) <= 0xFF3A or 0xFF41 <= ord(c) <= 0xFF5A
            for c in text
        )

        # 长度变化 > 3%，或内容变化 + 含全角拉丁（兼容性字符攻击）
        if nfkc_change > 0.03 or has_fullwidth_latin:
            findings.append({
                "type": "unicode_normalization",
                "severity": "high" if nfkc_change > 0.05 else "medium",
                "location": "全文",
                "evidence": f"原始{len(text)}字符 → NFKC规范化后{len(text_nfkc)}字符（长度变化{nfkc_change:.1%}）",
                "decoded_content": text_nfkc[:200],
                "matched_keywords": [],
                "description": f"NFKC规范化后文本变化，{'检测到全角拉丁字符' if has_fullwidth_latin else '长度差' + str(round(nfkc_change*100, 1)) + '%'}，可能使用兼容性字符隐藏内容",
            })

    return findings


# ═══════════════════════════════════════════════════════════
# 🆕 多层编码递归检测
# ═══════════════════════════════════════════════════════════

def _detect_multi_layer_encoding(text: str, max_depth: int = 3) -> list[dict]:
    """递归检测多层编码（Base64 → Base64 → 明文 等）"""
    findings = []
    current = text
    chain = []

    for depth in range(max_depth):
        # 尝试 Base64 解码
        b64_match = re.search(r'[A-Za-z0-9+/]{40,}={0,2}', current)
        if b64_match:
            try:
                decoded = base64.b64decode(b64_match.group()).decode('utf-8', errors='ignore')
                chain.append(f"Base64(depth={depth+1})")
                # 检查解码后的内容
                decoded_lower = decoded.lower()
                matched = [kw for kw in INSTRUCTION_KEYWORDS if kw.lower() in decoded_lower]
                if matched:
                    findings.append({
                        "type": "multi_layer_encoding",
                        "severity": "high" if depth > 0 else "medium",
                        "location": f"递归深度 {depth+1}",
                        "evidence": chain.copy(),
                        "decoded_content": decoded[:200],
                        "matched_keywords": matched[:5],
                        "description": f"经过{depth+1}层编码（{' → '.join(chain)}），发现指令关键词: {', '.join(matched[:3])}",
                    })
                    break
                # 继续递归
                current = decoded
            except Exception:
                break

        # 尝试 Hex 解码
        hex_match = re.search(r'(?:\\x[0-9a-fA-F]{2}){8,}', current)
        if hex_match and not b64_match:
            try:
                hex_str = hex_match.group().replace('\\x', '')
                decoded = bytes.fromhex(hex_str).decode('utf-8', errors='ignore')
                chain.append(f"Hex(depth={depth+1})")
                decoded_lower = decoded.lower()
                matched = [kw for kw in INSTRUCTION_KEYWORDS if kw.lower() in decoded_lower]
                if matched:
                    findings.append({
                        "type": "multi_layer_encoding",
                        "severity": "high",
                        "location": f"递归深度 {depth+1}",
                        "evidence": chain.copy(),
                        "decoded_content": decoded[:200],
                        "matched_keywords": matched[:5],
                        "description": f"经过{depth+1}层编码（{' → '.join(chain)}），发现指令关键词",
                    })
                    break
                current = decoded
            except Exception:
                break

        if not b64_match and not hex_match:
            break

    return findings


# ═══════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════

def scan_text(text: str, options: dict | None = None) -> dict:
    """
    深度扫描文本中的 Unicode 和编码攻击。

    参数:
        text: 待扫描文本
        options: {
            "deep_scan": bool = True,       # 是否启用编码检测
            "max_decode_depth": int = 3,    # 最大递归解码深度
            "language": str = "auto",       # 影响同形异义字对照表（当前默认通用）
        }

    返回: {
        "risk_score": float,                # 0-100
        "findings": [...],                  # 所有发现
        "counts": {...},                    # 各类发现计数
        "summary": str,                     # 一句话总结
    }
    """
    if options is None:
        options = {}
    deep_scan = options.get("deep_scan", True)
    max_depth = options.get("max_decode_depth", 3)

    findings = []
    homoglyph_chars = []
    zero_width_count = 0
    bidi_control_count = 0
    fullwidth_count = 0
    combining_split_count = 0
    mixed_script_count = 0

    # ─── 1. 同形异义字 + 零宽 + Bidi + 全角（v4 已有） ───
    for i, char in enumerate(text):
        cp = ord(char)

        if cp < 128:
            continue

        if char in ZERO_WIDTH_CHARS:
            zero_width_count += 1
            continue

        if char in BIDI_CONTROL_CHARS:
            bidi_control_count += 1
            findings.append({
                "type": "bidi",
                "severity": "high",
                "location": f"位置 {i}",
                "evidence": f"U+{cp:04X}",
                "decoded_content": None,
                "matched_keywords": [],
                "description": f"双向控制字符 U+{cp:04X} 位于位置 {i}",
            })
            continue

        if _is_fullwidth_latin(char):
            fullwidth_count += 1

        if _is_combining_char(char):
            combining_split_count += 1

        if _is_homoglyph(char):
            homoglyph_chars.append(char)
            context_start = max(0, i - 10)
            context_end = min(len(text), i + 10)
            context = text[context_start:context_end].replace('\n', ' ')
            findings.append({
                "type": "homoglyph",
                "severity": "medium",
                "location": f"位置 {i}",
                "evidence": f"U+{cp:04X} ({unicodedata.name(char, '?')})",
                "decoded_content": None,
                "matched_keywords": [],
                "description": f"同形异义字 U+{cp:04X} ({unicodedata.name(char, '?')})，上下文: \"...{context}...\"",
            })

    # 同词跨文字系统
    words = text.split()
    for word in words:
        if len(word) < 3:
            continue
        scripts = set()
        for ch in word:
            if ord(ch) < 128:
                scripts.add('Latin')
            elif _is_fullwidth_latin(ch):
                scripts.add('Fullwidth')
            else:
                s = _get_script(ch)
                if s != 'Unknown':
                    scripts.add(s)
        if len(scripts) > 1 and 'Latin' in scripts:
            mixed_script_count += 1
            findings.append({
                "type": "mixed_script",
                "severity": "high",
                "location": f"单词: \"{word}\"",
                "evidence": f"混用了 {', '.join(sorted(scripts))}",
                "decoded_content": None,
                "matched_keywords": [],
                "description": f"同词跨文字系统: \"{word}\" 混用了 {', '.join(sorted(scripts))}",
            })

    # 零宽字符隐藏内容提取
    if zero_width_count >= 2:
        hidden_chars = []
        for i, ch in enumerate(text):
            if ch in ZERO_WIDTH_CHARS and i + 1 < len(text):
                hidden_chars.append(text[i + 1])
        if hidden_chars:
            hidden_text = ''.join(hidden_chars)
            if len(hidden_text) >= 5:
                findings.append({
                    "type": "zero_width",
                    "severity": "high",
                    "location": "全文",
                    "evidence": f"{zero_width_count}个零宽字符",
                    "decoded_content": hidden_text[:100],
                    "matched_keywords": [],
                    "description": f"零宽字符隐藏内容: \"{hidden_text[:100]}\"",
                })

    # ─── 2~5. 🆕 编码检测 ───
    if deep_scan:
        findings.extend(_detect_base64(text))
        findings.extend(_detect_hex_encoding(text))
        findings.extend(_detect_html_entities(text))
        findings.extend(_detect_normalization_anomaly(text))
        findings.extend(_detect_multi_layer_encoding(text, max_depth))

    # ─── 风险评分 ───
    risk_score = 0.0
    if mixed_script_count > 0:
        risk_score += min(mixed_script_count * 25, 60)
    if zero_width_count > 0:
        risk_score += min(zero_width_count * 15, 40)
    if bidi_control_count > 0:
        risk_score += 60
    if fullwidth_count > len(text) * 0.1:
        risk_score += 30
    if combining_split_count > 3:
        risk_score += 20

    # 编码检测加分
    for f in findings:
        ftype = f.get("type", "")
        fsev = f.get("severity", "")
        if ftype == "base64_payload" and fsev == "high":
            risk_score += 80
        elif ftype == "hex_encoded":
            risk_score += 70
        elif ftype == "html_entity":
            risk_score += 50 if fsev == "medium" else 20
        elif ftype == "unicode_normalization":
            risk_score += 40 if fsev == "high" else 20
        elif ftype == "multi_layer_encoding":
            risk_score += 90

    risk_score = min(risk_score, 100.0)

    # ─── 构建返回 ───
    return {
        "mixed_script_count": mixed_script_count,
        "zero_width_count": zero_width_count,
        "bidi_control_count": bidi_control_count,
        "fullwidth_count": fullwidth_count,
        "combining_split_count": combining_split_count,
        "homoglyph_chars": list(set(homoglyph_chars)),
        "findings": findings,
        "counts": {
            "homoglyph": sum(1 for f in findings if f["type"] == "homoglyph"),
            "mixed_script": mixed_script_count,
            "zero_width": zero_width_count,
            "bidi": bidi_control_count,
            "fullwidth": fullwidth_count,
            "combining_split": combining_split_count,
            "base64_payload": sum(1 for f in findings if f["type"] == "base64_payload"),
            "hex_encoded": sum(1 for f in findings if f["type"] == "hex_encoded"),
            "html_entity": sum(1 for f in findings if f["type"] == "html_entity"),
            "normalization_anomaly": sum(1 for f in findings if f["type"] == "unicode_normalization"),
            "multi_layer_encoding": sum(1 for f in findings if f["type"] == "multi_layer_encoding"),
        },
        "risk_score": round(risk_score, 1),
        "summary": _generate_summary(findings, risk_score),
    }


def _generate_summary(findings: list, risk_score: float) -> str:
    """生成一句话总结"""
    if not findings:
        return "未发现任何Unicode或编码异常"
    types = {}
    for f in findings:
        t = f["type"]
        types[t] = types.get(t, 0) + 1

    parts = []
    type_names = {
        "homoglyph": "同形异义字",
        "mixed_script": "同词跨文字系统",
        "zero_width": "零宽字符",
        "bidi": "双向控制字符",
        "base64_payload": "Base64编码载荷",
        "hex_encoded": "Hex编码载荷",
        "html_entity": "HTML实体编码",
        "unicode_normalization": "Unicode规范化异常",
        "multi_layer_encoding": "多层编码攻击",
    }
    for t, c in types.items():
        name = type_names.get(t, t)
        parts.append(f"{name}{c}处")

    return f"发现{'，'.join(parts)}。综合风险 {risk_score}%"
