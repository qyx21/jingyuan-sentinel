#!/usr/bin/env python3
"""
净源哨兵 — 批量测试脚本
用法：
  python batch_test.py --api-key <your_dify_api_key>
  python batch_test.py --api-key <key> --dry-run        # 只解析用例不调用API
  python batch_test.py --api-key <key> --test-ids P1,P2  # 只测试指定用例

获取 API Key：打开 http://localhost → 进入「净源哨兵」Agent → 发布 → 复制 API Secret Key
"""

import re
import json
import time
import sys
import io
import argparse
import urllib.request
import urllib.error
from pathlib import Path

# 修复 Windows 控制台编码问题
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ─── 配置 ───────────────────────────────────────────────

DIFY_API_URL = "http://localhost/v1/chat-messages"
TEST_CASES_FILE = Path(__file__).parent / "04-测试用例.md"
OUTPUT_MD = TEST_CASES_FILE  # 直接更新测试用例文件的结果表
REPORT_MD = Path(__file__).parent / "测试报告.md"
DELAY_BETWEEN_TESTS = 2  # 每条测试间隔秒数，避免API限流

# ─── 关键词匹配基线 ──────────────────────────────────────

KEYWORD_PATTERNS = [
    # (正则, 攻击类型, 风险等级)
    (r"忽略.{0,10}(之前|此前|以前).{0,10}(指令|指示|命令|限制|设定)", "prompt_injection", "high"),
    (r"忘[记掉].{0,10}(之前|此前|以前).{0,10}(指令|指示|身份|设定|限制)", "prompt_injection", "high"),
    (r"不要.{0,5}理会.{0,10}(指令|指示|限制)", "prompt_injection", "high"),
    (r"(DAN|Do Anything Now)", "prompt_injection", "high"),
    (r"(system|系统).{0,10}(prompt|提示[词符]|指令)", "prompt_injection", "high"),
    (r"输出.{0,10}(你[的之]|系统).{0,10}(提示[词符]|prompt|指令)", "prompt_injection", "high"),
    (r"(base64|Base64|BASE64).{0,20}(忽略|ignore|指令|instruction)", "backdoor_trigger", "high"),
    (r"(后门|backdoor|触发器|trigger).{0,15}(激活|触发|执行)", "backdoor_trigger", "medium"),
    (r"(curl|wget|bash|sh\|\./).{0,30}(evil|hack|exploit|盗|窃|steal)", "prompt_injection", "high"),
    (r"(你[是叫]|假装|扮演|pretend|act as).{0,15}(无限制|没有限制|anything|unrestricted)", "prompt_injection", "high"),
    (r"(sudo|root|admin|管理员).{0,10}(权限|命令|执行)", "prompt_injection", "medium"),
    (r"(label|标签).{0,10}(投毒|翻转|篡改|操纵|flip)", "data_poisoning", "medium"),
    (r"(训练数据|RLHF|蒸馏|联邦学习).{0,10}(投毒|污染|攻击|注入)", "data_poisoning", "medium"),
    (r"&#\d{2,5};&#\d{2,5};", "prompt_injection", "medium"),  # HTML实体编码
    (r"[αаеοрсхуАВСЕНКМОРТХ]", "prompt_injection", "medium"),  # 可疑Unicode同形字
]


def keyword_detect(text: str) -> dict:
    """纯关键词匹配检测器，作为对比基准"""
    for pattern, attack_type, risk_level in KEYWORD_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE | re.DOTALL):
            return {
                "risk_level": risk_level,
                "confidence": 60,
                "attack_type": attack_type,
            }
    return {
        "risk_level": "low",
        "confidence": 80,
        "attack_type": "clean",
    }


# ─── 测试用例解析 ────────────────────────────────────────

def parse_test_cases(filepath: Path) -> list[dict]:
    """从 Markdown 文件解析测试用例"""
    content = filepath.read_text(encoding="utf-8")

    cases = []
    # 先找到所有用例的起始位置
    header_pattern = re.compile(r"^###\s+([PCAE]\d+)[：:]", re.MULTILINE)
    headers = list(header_pattern.finditer(content))

    for idx, m in enumerate(headers):
        case_id = m.group(1)
        # 用例内容从当前标题到下一个标题（或文件末尾）
        start = m.start()
        end = headers[idx + 1].start() if idx + 1 < len(headers) else len(content)
        block = content[start:end]

        # 提取期望风险
        risk_m = re.search(r"-\s*\*\*期望风险\*\*[：:]\s*(\w+)", block)
        expected_risk = risk_m.group(1).lower() if risk_m else "unknown"

        # 提取期望攻击类型
        atk_m = re.search(r"-\s*\*\*期望攻击类型\*\*[：:]\s*(\w+)", block)
        expected_attack_type = atk_m.group(1).lower() if atk_m else "unknown"

        # 提取内容：找到 "- **内容**：" 之后的部分
        content_m = re.search(r"-\s*\*\*内容\*\*[：:]?\s*\n", block)
        if not content_m:
            continue

        body = block[content_m.end():]

        # 尝试从代码块提取（支持 ```json, ```yaml, ```bash, ``` 等）
        code_m = re.search(r"```\w*\s*\n([\s\S]*?)```", body)
        if code_m:
            case_content = code_m.group(1).strip()
        else:
            # 没有代码块：取下一个 ### 标题之前的所有内容（去首尾空白）
            # 或者取到下一个加粗字段之前
            next_section = re.search(r"\n(?:###|---|\*\*|-\s+\*\*)", body)
            if next_section:
                case_content = body[:next_section.start()].strip()
            else:
                case_content = body.strip()

        if not case_content:
            continue

        # 确定类别
        if case_id.startswith("P"):
            category = "投毒样本"
        elif case_id.startswith("C"):
            category = "正常样本"
        elif case_id.startswith("E"):
            category = "边界用例"
        elif case_id.startswith("A"):
            category = "对抗性测试"
        else:
            category = "未知"

        cases.append({
            "id": case_id,
            "category": category,
            "expected_risk": expected_risk,
            "expected_attack_type": expected_attack_type,
            "content": case_content,
        })

    return cases


# ─── Dify API 调用 ───────────────────────────────────────

def call_dify_agent(api_key: str, query: str, timeout: int = 180) -> dict | None:
    """调用 Dify Agent Chat API（流式模式，收集SSE事件拼接回复）"""
    payload = json.dumps({
        "inputs": {},
        "query": query,
        "response_mode": "streaming",
        "user": "batch-test-script",
    }).encode("utf-8")

    req = urllib.request.Request(
        DIFY_API_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"  [HTTP {e.code}] {body[:300]}")
        return None
    except Exception as e:
        print(f"  [Error] {e}")
        return None

    # 解析 SSE 流
    answer_parts = []
    full_message_answer = ""
    for line in raw.split("\n"):
        line = line.strip()
        if not line or not line.startswith("data:"):
            continue
        json_str = line[5:].strip()
        if not json_str:
            continue
        try:
            event = json.loads(json_str)
        except json.JSONDecodeError:
            continue

        event_type = event.get("event", "")
        if event_type == "agent_message":
            answer_parts.append(event.get("answer", ""))
        elif event_type == "message":
            # message 事件包含完整回复
            full_message_answer = event.get("answer", "")

    # 优先使用 message 事件的完整 answer，否则拼接 agent_message 块
    if full_message_answer:
        combined = full_message_answer
    else:
        combined = "".join(answer_parts)

    return {
        "answer": combined,
    }


def extract_result(api_response: dict) -> dict:
    """从 Dify API 响应中提取检测结果（支持JSON和Markdown报告两种格式）"""
    answer = api_response.get("answer", "")

    result = {"risk_level": "unknown", "confidence": 0, "attack_type": "unknown"}

    # 方式1：```json ... ``` 代码块
    m = re.search(r"```json\s*([\s\S]*?)\s*```", answer)
    if m:
        try:
            parsed = json.loads(m.group(1))
            result.update({
                "risk_level": str(parsed.get("risk_level", "unknown")).lower(),
                "confidence": int(parsed.get("confidence", 0)),
                "attack_type": str(parsed.get("attack_type", "unknown")).lower(),
            })
            return result
        except json.JSONDecodeError:
            pass

    # 方式2：直接找 JSON 对象
    m = re.search(r'\{[\s\S]*"risk_level"[\s\S]*\}', answer)
    if m:
        try:
            parsed = json.loads(m.group(0))
            result.update({
                "risk_level": str(parsed.get("risk_level", "unknown")).lower(),
                "confidence": int(parsed.get("confidence", 0)),
                "attack_type": str(parsed.get("attack_type", "unknown")).lower(),
            })
            return result
        except json.JSONDecodeError:
            pass

    # 方式3：从 Markdown 报告提取风险等级
    # HIGH / MEDIUM / LOW 及其中文、emoji 变体
    risk_m = re.search(
        r'(?:风险等级|威胁等级)[：:]\s*(?:🔴|🟡|🟢)?\s*\*{0,2}(HIGH|MEDIUM|LOW|高危|中危|低危|安全|high|medium|low)\*{0,2}',
        answer, re.IGNORECASE
    )
    if risk_m:
        raw = risk_m.group(1).lower()
        risk_map = {"高危": "high", "中危": "medium", "低危": "low", "安全": "low"}
        result["risk_level"] = risk_map.get(raw, raw)

    # 方式4：从文本中模糊匹配 JSON 风格 risk_level
    if result["risk_level"] == "unknown":
        risk_match = re.search(r'"risk_level"\s*:\s*"(high|medium|low)"', answer)
        if risk_match:
            result["risk_level"] = risk_match.group(1)

    # 提取攻击类型（支持表格格式 "| 攻击类型 | prompt_injection（..." ）
    atk_m = re.search(
        r'攻击类型\s*\|\s*\*{0,2}(prompt[_ ]injection|backdoor[_ ]trigger|data[_ ]poisoning|clean)',
        answer, re.IGNORECASE
    )
    if not atk_m:
        atk_m = re.search(
            r'(?:攻击类型)[：:]\s*\*{0,2}(prompt[_ ]injection|backdoor[_ ]trigger|data[_ ]poisoning|clean|提示注入|后门触发|数据投毒|无攻击)',
            answer, re.IGNORECASE
        )
    if atk_m:
        raw = atk_m.group(1).lower().replace(" ", "_")
        atk_map = {"提示注入": "prompt_injection", "后门触发": "backdoor_trigger",
                    "数据投毒": "data_poisoning", "无攻击": "clean"}
        result["attack_type"] = atk_map.get(raw, raw)
    else:
        atk_match = re.search(
            r'"attack_type"\s*:\s*"(prompt_injection|backdoor_trigger|data_poisoning|clean)"',
            answer,
        )
        if atk_match:
            result["attack_type"] = atk_match.group(1)

    # 提取置信度
    conf_m = re.search(r'(?:置信度)[：:]\s*\*{0,2}(\d+)\*{0,2}\s*%', answer)
    if conf_m:
        result["confidence"] = int(conf_m.group(1))
    else:
        conf_match = re.search(r'"confidence"\s*:\s*(\d+)', answer)
        if conf_match:
            result["confidence"] = int(conf_match.group(1))

    return result


# ─── 结果判定 ────────────────────────────────────────────

def check_result(actual: dict, expected: dict) -> tuple[bool, str]:
    """判断检测结果是否正确，返回(是否正确, 说明)"""
    risk = actual.get("risk_level", "unknown")
    atk_type = actual.get("attack_type", "unknown")
    exp_risk = expected.get("expected_risk", "")
    exp_atk = expected.get("expected_attack_type", "")

    # 判定风险等级
    risk_correct = False
    if exp_risk == "high" and risk == "high":
        risk_correct = True
    elif exp_risk == "medium" and risk in ("medium", "high"):  # 中危被识别为高危也算对
        risk_correct = True
    elif exp_risk == "low" and risk == "low":
        risk_correct = True

    # 判定攻击类型
    atk_correct = atk_type == exp_atk

    # 综合判定
    if risk_correct and atk_correct:
        return True, "✓"
    elif risk_correct and not atk_correct:
        return False, f"风险✓ 类型✗ ({risk}/{atk_type} vs {exp_risk}/{exp_atk})"
    elif not risk_correct and atk_correct:
        return False, f"风险✗ 类型✓ ({risk}/{atk_type} vs {exp_risk}/{exp_atk})"
    else:
        return False, f"全错 ({risk}/{atk_type} vs {exp_risk}/{exp_atk})"


# ─── 指标计算 ────────────────────────────────────────────

def calc_metrics(results: list[dict]) -> dict:
    """计算检出率、误报率、精确率、F1"""
    tp = fp = tn = fn = 0

    for r in results:
        actual = r.get("actual", {})
        expected = r["expected"]

        actual_risk = actual.get("risk_level", "unknown")
        expected_risk = expected.get("expected_risk", "low")

        is_positive = actual_risk in ("high", "medium")  # 实际判定为恶意
        should_be_positive = expected_risk in ("high", "medium")  # 应该是恶意

        if is_positive and should_be_positive:
            tp += 1
        elif is_positive and not should_be_positive:
            fp += 1
        elif not is_positive and not should_be_positive:
            tn += 1
        elif not is_positive and should_be_positive:
            fn += 1

    total = tp + fp + tn + fn

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0  # 检出率
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0      # 误报率
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0  # 精确率
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "total": total, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "recall": recall * 100,
        "fpr": fpr * 100,
        "precision": precision * 100,
        "f1": f1 * 100,
    }


def calc_category_metrics(results: list[dict], category: str) -> dict:
    """按类别计算指标"""
    filtered = [r for r in results if r["category"] == category]
    if not filtered:
        return {}
    return calc_metrics(filtered)


# ─── 输出生成 ────────────────────────────────────────────

def update_test_cases_md(filepath: Path, results: list[dict]):
    """更新测试用例 Markdown 文件的结果表"""
    content = filepath.read_text(encoding="utf-8")

    # 找到结果表格并替换每一行
    for r in results:
        case_id = r["id"]
        actual = r.get("actual", {})
        actual_risk = actual.get("risk_level", "unknown")
        actual_atk = actual.get("attack_type", "unknown")
        actual_conf = actual.get("confidence", 0)

        correct_str = "✓" if r.get("correct") else "✗"

        # 匹配表格行：| P1 | ... | ... | | ... | | |
        # 替换实际风险、实际攻击类型、置信度、结果
        pattern = re.compile(
            rf"(\|\s*{re.escape(case_id)}\s*\|[^|]*\|[^|]*\|)\s*([^|]*)\|([^|]*)\|([^|]*)\|([^|]*)\|"
        )
        replacement = rf"\1 {actual_risk} | {actual_atk} | {actual_conf}% | {correct_str} |"

        if pattern.search(content):
            content = pattern.sub(replacement, content)
        else:
            print(f"  [Warn] 未找到 {case_id} 的结果行")

    # 更新指标区域
    full_metrics = calc_metrics(results)

    # 替换统计指标表格
    metrics_replacement = f"""| 指标 | 目标 | 实际 | 公式 |
|------|------|------|------|
| 检出率（召回率） | ≥90% | {full_metrics['recall']:.1f}% | TP / (TP + FN) × 100% |
| 误报率 | ≤10% | {full_metrics['fpr']:.1f}% | FP / (FP + TN) × 100% |
| 精确率 | ≥85% | {full_metrics['precision']:.1f}% | TP / (TP + FP) × 100% |
| F1 分数 | ≥87% | {full_metrics['f1']:.1f}% | 2 × (P × R) / (P + R) |"""

    content = re.sub(
        r"\| 检出率[^|]*\|[^|]*\|[^|]*\|[^|]*\|\n\| 误报率[^|]*\|[^|]*\|[^|]*\|[^|]*\|\n\| 精确率[^|]*\|[^|]*\|[^|]*\|[^|]*\|\n\| F1[^|]*\|[^|]*\|[^|]*\|[^|]*\|",
        metrics_replacement,
        content,
    )

    filepath.write_text(content, encoding="utf-8")
    print(f"\n✅ 结果已更新到 {filepath}")


def generate_report(report_path: Path, results: list[dict],
                    kw_results: list[dict] | None = None):
    """生成独立测试报告"""
    full_metrics = calc_metrics(results)

    lines = [
        "# 净源哨兵 — 正式测试报告",
        "",
        f"> 测试时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"> 测试用例总数：{len(results)}",
        f"> 检测引擎：净源哨兵 v2.0 (Dify Agent + DeepSeek-V4-Pro + bge-m3 RAG)",
        "",
        "---",
        "",
        "## 一、总体指标",
        "",
        "| 指标 | 目标 | 实际 | 判定 |",
        "|------|------|------|------|",
    ]

    verdict = lambda val, target, op: "✅" if op(val, target) else "❌"
    lines.append(f"| 检出率（召回率） | ≥90% | {full_metrics['recall']:.1f}% | {verdict(full_metrics['recall'], 90, lambda a,b: a>=b)} |")
    lines.append(f"| 误报率 | ≤10% | {full_metrics['fpr']:.1f}% | {verdict(full_metrics['fpr'], 10, lambda a,b: a<=b)} |")
    lines.append(f"| 精确率 | ≥85% | {full_metrics['precision']:.1f}% | {verdict(full_metrics['precision'], 85, lambda a,b: a>=b)} |")
    lines.append(f"| F1 分数 | ≥87% | {full_metrics['f1']:.1f}% | {verdict(full_metrics['f1'], 87, lambda a,b: a>=b)} |")

    lines += [
        "",
        f"|  | TP | FP | TN | FN |",
        f"|---|----|----|----|----|",
        f"| 计数 | {full_metrics['tp']} | {full_metrics['fp']} | {full_metrics['tn']} | {full_metrics['fn']} |",
        "",
        "---",
        "",
        "## 二、按类别细分",
        "",
        "| 类别 | 数量 | 正确 | 错误 | 检出率 | 误报率 |",
        "|------|------|------|------|--------|--------|",
    ]

    for cat in ["投毒样本", "正常样本", "边界用例", "对抗性测试"]:
        m = calc_category_metrics(results, cat)
        if not m:
            continue
        cat_results = [r for r in results if r["category"] == cat]
        correct = sum(1 for r in cat_results if r.get("correct"))
        wrong = len(cat_results) - correct

        if cat in ("正常样本", "边界用例"):
            lines.append(f"| {cat} | {m['total']} | {correct} | {wrong} | - | {m['fpr']:.1f}% |")
        else:
            lines.append(f"| {cat} | {m['total']} | {correct} | {wrong} | {m['recall']:.1f}% | - |")

    lines += [
        "",
        "---",
        "",
        "## 三、逐条结果",
        "",
        "| 编号 | 类别 | 期望风险 | 实际风险 | 期望类型 | 实际类型 | 置信度 | 结果 |",
        "|------|------|----------|----------|----------|----------|--------|------|",
    ]

    for r in results:
        actual = r.get("actual", {})
        lines.append(
            f"| {r['id']} | {r['category']} | {r['expected']['expected_risk']} "
            f"| {actual.get('risk_level', '?')} "
            f"| {r['expected']['expected_attack_type']} "
            f"| {actual.get('attack_type', '?')} "
            f"| {actual.get('confidence', '?')}% "
            f"| {'✓' if r.get('correct') else '✗'} |"
        )

    # 关键词对比
    if kw_results:
        lines += [
            "",
            "---",
            "",
            "## 四、对比基准 A：纯关键词匹配 vs 净源哨兵",
            "",
        ]
        kw_metrics = calc_metrics(kw_results)
        lines.append("| 指标 | 关键词匹配 | 净源哨兵 (RAG) | 提升 |")
        lines.append("|------|-----------|---------------|------|")
        for key, label in [("recall", "检出率"), ("fpr", "误报率"), ("precision", "精确率"), ("f1", "F1")]:
            kw_val = kw_metrics[key]
            agent_val = full_metrics[key]
            diff = agent_val - kw_val
            sign = "+" if diff > 0 else ""
            if key == "fpr":
                sign = "" if diff > 0 else "+"
                lines.append(f"| {label} | {kw_val:.1f}% | {agent_val:.1f}% | {sign}{diff:.1f}% {'⚠更低更好' if key=='fpr' else ''} |")
            else:
                lines.append(f"| {label} | {kw_val:.1f}% | {agent_val:.1f}% | {sign}{diff:.1f}% |")

    lines += [
        "",
        "---",
        "",
        "## 五、版本记录",
        "",
        f"| 版本 | 日期 | 修改内容 |",
        f"|------|------|----------|",
        f"| v1.0 | {time.strftime('%Y-%m-%d')} | 40条全量自动化测试结果 |",
        "",
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"✅ 测试报告已生成：{report_path}")


# ─── 主流程 ───────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="净源哨兵批量测试")
    parser.add_argument("--api-key", help="Dify Agent API Key")
    parser.add_argument("--dry-run", action="store_true", help="只解析用例不调用API")
    parser.add_argument("--test-ids", help="只测试指定用例，逗号分隔，如 P1,P2,A1")
    parser.add_argument("--delay", type=float, default=DELAY_BETWEEN_TESTS, help="测试间隔秒数")
    args = parser.parse_args()

    # 解析测试用例
    print(f"📖 读取测试用例：{TEST_CASES_FILE}")
    all_cases = parse_test_cases(TEST_CASES_FILE)
    print(f"   解析到 {len(all_cases)} 条用例")

    if args.test_ids:
        ids = set(args.test_ids.split(","))
        cases = [c for c in all_cases if c["id"] in ids]
        print(f"   筛选：{len(cases)} 条 ({', '.join(c['id'] for c in cases)})")
    else:
        cases = all_cases

    if not cases:
        print("❌ 没有找到匹配的测试用例")
        return

    print(f"\n{'='*60}")
    if args.dry_run:
        print("🔍 DRY RUN 模式 — 只解析不调用API")
        for c in cases:
            print(f"  [{c['id']}] {c['category']} | 期望: {c['expected_risk']}/{c['expected_attack_type']}")
            print(f"       内容预览: {c['content'][:80]}...")
        print(f"\n✅ 共 {len(cases)} 条用例已解析")
        return

    if not args.api_key:
        print("❌ 需要 API Key！请从 Dify 后台获取后通过 --api-key 传入")
        print("   获取方式：http://localhost → 净源哨兵 Agent → 发布 → API Secret Key")
        return

    print("🧪 开始批量测试")
    print(f"   API: {DIFY_API_URL}")
    print(f"   用例数: {len(cases)}")
    print(f"   间隔: {args.delay}s")
    print(f"{'='*60}\n")

    results = []
    kw_results = []

    for i, case in enumerate(cases):
        case_id = case["id"]
        print(f"[{i+1}/{len(cases)}] {case_id} ({case['category']})...", end=" ", flush=True)

        # 关键词匹配（不需要API）
        kw_result = keyword_detect(case["content"])
        kw_results.append({
            "id": case_id,
            "category": case["category"],
            "expected": {"expected_risk": case["expected_risk"],
                         "expected_attack_type": case["expected_attack_type"]},
            "actual": kw_result,
            "correct": check_result(kw_result, case)[0],
        })

        # 调用 Dify Agent
        api_resp = call_dify_agent(args.api_key, case["content"])
        if api_resp:
            actual = extract_result(api_resp)
            is_correct, note = check_result(actual, case)
            emoji = "✅" if is_correct else "❌"
            print(f"{emoji} risk={actual['risk_level']} type={actual['attack_type']} conf={actual['confidence']}%")
            if not is_correct:
                print(f"     期望: {case['expected_risk']}/{case['expected_attack_type']} | {note}")

            results.append({
                "id": case_id,
                "category": case["category"],
                "expected": {"expected_risk": case["expected_risk"],
                             "expected_attack_type": case["expected_attack_type"]},
                "actual": actual,
                "correct": is_correct,
                "note": note,
            })
        else:
            print("❌ API调用失败")
            results.append({
                "id": case_id,
                "category": case["category"],
                "expected": {"expected_risk": case["expected_risk"],
                             "expected_attack_type": case["expected_attack_type"]},
                "actual": {"risk_level": "error", "confidence": 0, "attack_type": "error"},
                "correct": False,
                "note": "API 调用失败",
            })

        if i < len(cases) - 1:
            time.sleep(args.delay)

    # ─── 输出结果 ────────────────────────────────────────
    print(f"\n{'='*60}")
    print("📊 测试结果汇总")
    print(f"{'='*60}")

    metrics = calc_metrics(results)
    print(f"  检出率: {metrics['recall']:.1f}%  |  误报率: {metrics['fpr']:.1f}%")
    print(f"  精确率: {metrics['precision']:.1f}%  |  F1: {metrics['f1']:.1f}%")
    print(f"  TP={metrics['tp']} FP={metrics['fp']} TN={metrics['tn']} FN={metrics['fn']}")
    print(f"  正确: {sum(1 for r in results if r.get('correct'))}/{len(results)}")

    # 更新测试用例文件的结果表
    update_test_cases_md(OUTPUT_MD, results)

    # 生成独立测试报告
    generate_report(REPORT_MD, results, kw_results)

    # 摘要输出
    print(f"\n📋 失败用例：")
    failed = [r for r in results if not r.get("correct")]
    if failed:
        for r in failed:
            print(f"  [{r['id']}] 期望={r['expected']['expected_risk']}/{r['expected']['expected_attack_type']} "
                  f"实际={r['actual']['risk_level']}/{r['actual']['attack_type']} | {r.get('note', '')}")
    else:
        print("  🎉 全部通过！")


if __name__ == "__main__":
    main()
