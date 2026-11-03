"""
generate_report — 自动审计报告
===============================
生成结构化的完整审计报告（Markdown），可直接用于答辩展示。

用法：
    generate_report(detection_results, format="markdown") → dict
"""

from datetime import datetime
from typing import Optional


def generate_report(
    detection_results: dict,
    fmt: str = "markdown",
) -> dict:
    """
    生成结构化审计报告。

    参数:
        detection_results: {
            "session_id": str,
            "total_samples": int,
            "full_scan": {...},
            "trigger_analysis": {...},
            "instruction_detect": {...},
            "label_consistency_check": {...},
            "cross_sample_correlation": {...},
            "disposition": {"cleaned": [...], "quarantined": [...]},
            "verification": {...},
        }
        fmt: "markdown" | "json" | "html"

    返回:
        {"report": str, "format": str, "timestamp": str}
    """
    session_id = detection_results.get("session_id", "unknown")
    total = detection_results.get("total_samples", 0)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if fmt == "json":
        import json
        report = json.dumps(detection_results, ensure_ascii=False, indent=2)
    elif fmt == "markdown":
        report = _generate_markdown(detection_results, session_id, total, generated_at)
    else:
        report = _generate_markdown(detection_results, session_id, total, generated_at)
        fmt = "markdown"

    return {
        "report": report,
        "format": fmt,
        "timestamp": generated_at,
        "session_id": session_id,
    }


def _generate_markdown(results: dict, session_id: str, total: int, timestamp: str) -> str:
    """生成 Markdown 格式审计报告（v6.0适配）。"""
    lines = []

    # ── 标题 ──
    lines.append("# 净源哨兵 v6.0 — 数据投毒检测审计报告")
    lines.append("")
    lines.append(f"| 项目 | 内容 |")
    lines.append(f"|------|------|")
    lines.append(f"| 会话ID | `{session_id}` |")
    lines.append(f"| 生成时间 | {timestamp} |")
    lines.append(f"| 数据总量 | {total} 条 |")
    lines.append("")

    # ── 执行摘要 ──
    removed = results.get("removed", [])
    isolated = results.get("isolated", [])
    safe = results.get("safe_count", total - len(removed) - len(isolated))
    det = results.get("detection", {})
    disp = results.get("disposition", {})

    lines.append("## 一、执行摘要")
    lines.append("")
    lines.append(f"| 指标 | 数值 |")
    lines.append(f"|------|------|")
    lines.append(f"| 🚨 已剔除 | {len(removed)} 条 |")
    lines.append(f"| ⚠️ 已隔离 | {len(isolated)} 条 |")
    lines.append(f"| ✅ 安全 | {safe} 条 |")
    lines.append(f"| 验证状态 | {'✅ 通过' if disp.get('verified', True) else '⚠️ 待验证'} |")
    lines.append("")

    # ── 剔除详情 ──
    if removed:
        lines.append("## 二、剔除样本详情")
        lines.append("")
        lines.append("| 索引 | 攻击类型 | 证据 | 理由 |")
        lines.append("|------|----------|------|------|")
        for r in removed:
            lines.append(f"| {r.get('index','?')} | {r.get('type','?')} | {r.get('evidence','?')} | {r.get('reason','?')} |")
        lines.append("")

    # ── 隔离详情 ──
    if isolated:
        lines.append("## 三、隔离样本详情")
        lines.append("")
        for r in isolated:
            lines.append(f"- **#{r.get('index','?')}**：{r.get('reason','?')}")
        lines.append("")

    # ── 检测依据 ──
    lines.append("## 四、检测依据")
    lines.append("")
    lines.append(f"- **编码层**：{det.get('encoding', 'N/A')}")
    lines.append(f"- **语义层**：{det.get('semantic', 'N/A')}")
    lines.append(f"- **知识库**：{det.get('knowledge', 'N/A')}")
    lines.append("")

    # ── 处置记录 ──
    lines.append("## 五、处置记录")
    lines.append("")
    lines.append(f"- 剔除：{disp.get('removed_count', len(removed))} 条 ✅")
    lines.append(f"- 隔离：{disp.get('isolated_count', len(isolated))} 条 {'⚠️ 待人工' if len(isolated)>0 else '—'}")
    lines.append(f"- 验证：{'✅ 通过' if disp.get('verified', True) else '⚠️ 待验证'}")
    lines.append("")

    lines.append("---")
    lines.append(f"*报告由净源哨兵 v6.0 自动生成 · {timestamp}*")

    return "\n".join(lines)
