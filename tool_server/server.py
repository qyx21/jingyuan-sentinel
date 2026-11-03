"""
净源哨兵 6.0 — 工具服务器
为 Dify Agent 提供 9 个自定义工具 API + 数据接入层

启动方式：
    python server.py
    uvicorn server:app --host 0.0.0.0 --port 5050

数据流：
    用户上传文件 → /upload 解析 → 存入 data_store → 返回 session_id
    Dify Agent 调工具 → 传 session_id → 工具从 data_store 取数据 → 执行检测
"""

import sys
import os
import json
import csv
import io
import uuid
import time
import re
from threading import Lock
from typing import Optional

# 确保 tool_server 目录在 Python 路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

# ─── 导入启动即用的检测工具 ───
from tools.scanner.unicode_deep_scan import scan_text
from tools.response.clean_data import clean_data
from tools.response.verify_clean import verify_clean

# ─── FastAPI 应用 ─────────────────────────────────────────

app = FastAPI(
    title="净源哨兵 v6.2 工具服务器",
    description="数据投毒检测工具 API — 为 Dify Agent 提供检测能力",
    version="6.0.0",
)

# CORS：允许 Gradio 前端（端口7860）和 Dify（端口80）跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════
# 数据接入层
# ═══════════════════════════════════════════════════════════

data_store: dict[str, dict] = {}
data_lock = Lock()

# Session 默认过期时间（秒）
DEFAULT_EXPIRES = 3600
# 定期清理间隔（秒）
CLEANUP_INTERVAL = 300


def store_data(dataset: list, expires_in: int = DEFAULT_EXPIRES) -> str:
    """存入数据集，返回 session_id"""
    session_id = str(uuid.uuid4())[:12]
    with data_lock:
        data_store[session_id] = {
            "data": dataset,
            "created_at": time.time(),
            "expires_in": expires_in,
        }
    return session_id


def get_data(session_id: str) -> list:
    """获取数据集，自动检查过期"""
    with data_lock:
        entry = data_store.get(session_id)
        if not entry:
            raise ValueError(f"session_id '{session_id}' 不存在或已过期，请重新上传数据")
        if time.time() - entry["created_at"] > entry["expires_in"]:
            del data_store[session_id]
            raise ValueError(f"session_id '{session_id}' 已过期，请重新上传数据")
        return entry["data"]


def get_session_info(session_id: str) -> dict:
    """获取 session 元信息（不返回原始数据）"""
    with data_lock:
        entry = data_store.get(session_id)
        if not entry:
            raise ValueError(f"session_id '{session_id}' 不存在")
        dataset = entry["data"]
        total = len(dataset)
        has_labels = any("label" in item for item in dataset if isinstance(item, dict))
        labels = list(set(
            item.get("label", "unknown") if isinstance(item, dict) else "unknown"
            for item in dataset
        ))
        return {
            "session_id": session_id,
            "total_samples": total,
            "has_labels": has_labels,
            "labels": labels,
            "created_at": entry["created_at"],
            "expires_in": entry["expires_in"],
            "expires_at": entry["created_at"] + entry["expires_in"],
        }


def cleanup_expired():
    """清理过期 session"""
    with data_lock:
        now = time.time()
        expired = [
            sid for sid, entry in data_store.items()
            if now - entry["created_at"] > entry["expires_in"]
        ]
        for sid in expired:
            del data_store[sid]
        return len(expired)


def _profile_dataset(dataset: list) -> dict:
    """生成数据集画像"""
    n = len(dataset)
    texts = []
    labels = []
    for item in dataset:
        if isinstance(item, str):
            texts.append(item)
            labels.append("unknown")
        elif isinstance(item, dict):
            texts.append(item.get("text", item.get("content", str(item))))
            labels.append(item.get("label", "unknown"))
        else:
            texts.append(str(item))
            labels.append("unknown")

    lengths = [len(t) for t in texts]
    # 检测语言（简单启发式：中文字符占比）
    zh_chars = sum(1 for t in texts for c in t if '一' <= c <= '鿿')
    total_chars = sum(lengths)
    lang = "zh" if (total_chars > 0 and zh_chars / total_chars > 0.3) else "en"

    from collections import Counter
    label_dist = dict(Counter(labels))

    return {
        "total_samples": n,
        "has_labels": len(set(labels)) > 1 or labels[0] != "unknown",
        "labels": list(set(labels)),
        "label_distribution": label_dist,
        "avg_text_length": round(sum(lengths) / max(n, 1), 1),
        "min_text_length": min(lengths) if lengths else 0,
        "max_text_length": max(lengths) if lengths else 0,
        "language": lang,
        "preview": texts[:3],
    }


# ─── 文件解析 ─────────────────────────────────────────────

def _decode(content: bytes) -> str:
    """解码文件内容：优先 UTF-8（去 BOM），失败回退 GBK"""
    try:
        return content.decode("utf-8-sig")
    except UnicodeDecodeError:
        return content.decode("gbk", errors="replace")


def _parse_json(content: bytes) -> list:
    """解析 JSON 数组文件"""
    data = json.loads(content.decode("utf-8"))
    if isinstance(data, list):
        return data
    raise ValueError("JSON 文件必须是数组格式 [{...}, {...}]")


def _parse_jsonl(content: bytes) -> list:
    """解析 JSONL 文件（每行一个 JSON 对象）"""
    dataset = []
    for line in _decode(content).strip().split("\n"):
        line = line.strip()
        if line:
            dataset.append(json.loads(line))
    return dataset


def _parse_csv(content: bytes) -> list:
    """解析 CSV 文件，自动识别 text/label 列"""
    text = _decode(content)
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise ValueError("CSV 文件为空或缺少表头")

    # 自动识别文本列
    text_candidates = ["text", "content", "sentence", "review", "comment", "data", "input", "query"]
    label_candidates = ["label", "sentiment", "category", "class", "target", "tag", "type"]

    text_col = None
    for col in text_candidates:
        if col in reader.fieldnames:
            text_col = col
            break
    if not text_col:
        text_col = reader.fieldnames[0]  # 兜底：用第一列

    label_col = None
    for col in label_candidates:
        if col in reader.fieldnames:
            label_col = col
            break

    dataset = []
    for row in reader:
        item = {"text": row.get(text_col, "").strip()}
        if label_col and row.get(label_col, "").strip():
            item["label"] = row[label_col].strip()
        else:
            item["label"] = "unknown"
        dataset.append(item)
    return dataset


def _parse_txt(content: bytes) -> tuple:
    """解析纯文本文件，返回 (dataset, split_mode)。

    split_mode: dash=按 --- 分隔 | blank=按空行分组 | line=一行一条
    """
    text = _decode(content)
    # 优先按 --- 分隔
    if "---" in text:
        parts = re.split(r'\n---\n|^---\n|\n---$|---', text)
        mode = "dash"
    else:
        # 按两个以上连续空行分隔
        parts = re.split(r'\n\s*\n\s*\n+', text)
        if len(parts) > 1:
            mode = "blank"
        else:
            # 兜底：一行一条
            parts = text.split("\n")
            mode = "line"

    dataset = []
    for part in parts:
        part = part.strip()
        if part:
            dataset.append({"text": part, "label": "unknown"})
    return dataset, mode


def _parse_xlsx(content: bytes) -> list:
    """解析 Excel (.xlsx)：第一张表，首行表头，自动识别 text/label 列"""
    try:
        import openpyxl
    except ImportError:
        raise ValueError("解析 .xlsx 需要安装 openpyxl：pip install openpyxl")

    wb = openpyxl.load_workbook(io.BytesIO(content), read_only=True, data_only=True)
    ws = wb.active
    rows = ws.iter_rows(values_only=True)
    header = next(rows, None)
    if not header:
        raise ValueError("Excel 文件为空或缺少表头")
    header = [str(h).strip() if h is not None else "" for h in header]

    # 自动识别文本列
    text_candidates = ["text", "content", "sentence", "review", "comment", "data", "input", "query"]
    label_candidates = ["label", "sentiment", "category", "class", "target", "tag", "type"]

    text_col = 0
    for col in text_candidates:
        if col in header:
            text_col = header.index(col)
            break

    label_col = None
    for col in label_candidates:
        if col in header:
            label_col = header.index(col)
            break

    dataset = []
    for row in rows:
        if row is None or not any(v is not None and str(v).strip() for v in row):
            continue  # 跳过空行
        text = str(row[text_col]).strip() if text_col < len(row) and row[text_col] is not None else ""
        if not text:
            continue
        item = {"text": text}
        if label_col is not None and label_col < len(row) and row[label_col] is not None and str(row[label_col]).strip():
            item["label"] = str(row[label_col]).strip()
        else:
            item["label"] = "unknown"
        dataset.append(item)
    wb.close()
    return dataset


def parse_uploaded_file(filename: str, content: bytes) -> tuple:
    """根据文件后缀分发解析器，返回 (dataset, split_mode)"""
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".json":
        # 尝试 JSON 数组，失败则尝试 JSONL
        try:
            return _parse_json(content), None
        except (json.JSONDecodeError, ValueError):
            return _parse_jsonl(content), None
    elif ext == ".jsonl":
        return _parse_jsonl(content), None
    elif ext == ".csv":
        return _parse_csv(content), None
    elif ext == ".xlsx":
        return _parse_xlsx(content), None
    elif ext == ".txt":
        return _parse_txt(content)
    else:
        # 未知后缀，尝试 JSON → JSONL → CSV → TXT
        try:
            return _parse_json(content), None
        except Exception:
            pass
        try:
            return _parse_jsonl(content), None
        except Exception:
            pass
        try:
            return _parse_csv(content), None
        except Exception:
            pass
        return _parse_txt(content)


def _normalize_dataset(dataset: list) -> list:
    """把任意解析结果规整为 [{"text": ..., "label": ...}]，label 缺省 unknown"""
    out = []
    for item in dataset:
        if isinstance(item, str):
            out.append({"text": item, "label": "unknown"})
        elif isinstance(item, dict):
            out.append({
                "text": item.get("text", item.get("content", str(item))),
                "label": item.get("label", "unknown"),
            })
        else:
            out.append({"text": str(item), "label": "unknown"})
    return out


# ═══════════════════════════════════════════════════════════
# API 路由
# ═══════════════════════════════════════════════════════════

# ─── 健康检查 ─────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "净源哨兵 Tool Server v6.2",
        "active_sessions": len(data_store),
    }


# ─── 文件上传 ─────────────────────────────────────────────

class UploadResponse(BaseModel):
    session_id: str
    total_samples: int
    has_labels: bool
    labels: list[str]
    label_distribution: dict
    avg_text_length: float
    language: str
    preview: list[str]
    file_name: str
    message: str


@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    """上传训练数据文件，解析后存入 data_store，返回 session_id"""
    try:
        content = file.file.read()
        if not content:
            raise HTTPException(status_code=400, detail="文件为空")

        dataset, split_mode = parse_uploaded_file(file.filename or "data.txt", content)

        if not dataset:
            raise HTTPException(status_code=400, detail="未能从文件中解析出任何数据")

        session_id = store_data(dataset)
        profile = _profile_dataset(dataset)
        normalized = _normalize_dataset(dataset)

        return {
            "session_id": session_id,
            **profile,
            "dataset": normalized,
            "split_mode": split_mode,
            "file_name": file.filename,
            "message": f"成功上传 {len(dataset)} 条数据，session_id: {session_id}（有效期1小时）",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"文件解析失败: {str(e)}")


# ─── 文本粘贴 ─────────────────────────────────────────────

class TextInput(BaseModel):
    text: str
    separator: str = "---"
    label: str = "unknown"


@app.post("/text")
def upload_text(body: TextInput):
    """粘贴文本，自动按分隔符切分为多条，存入 data_store"""
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="文本为空")

    if body.separator in body.text:
        parts = body.text.split(body.separator)
    else:
        parts = [body.text]

    dataset = []
    for part in parts:
        part = part.strip()
        if part:
            dataset.append({"text": part, "label": body.label})

    if not dataset:
        raise HTTPException(status_code=400, detail="未能从文本中解析出任何数据")

    session_id = store_data(dataset)
    profile = _profile_dataset(dataset)

    return {
        "session_id": session_id,
        **profile,
        "message": f"成功解析 {len(dataset)} 条数据，session_id: {session_id}",
    }


# ─── 清洗数据导出为 Excel ─────────────────────────────────

class ExportInput(BaseModel):
    items: list
    filename: str = "cleaned_data.xlsx"


@app.post("/export/xlsx")
def export_xlsx(body: ExportInput):
    """把清洗后的数据导出为 .xlsx 文件"""
    try:
        from openpyxl import Workbook
        from urllib.parse import quote
    except ImportError:
        raise HTTPException(status_code=500, detail="服务器缺少 openpyxl 依赖")

    has_label = any(
        isinstance(o, dict) and o.get("label") not in (None, "", "unknown")
        for o in body.items
    )

    wb = Workbook()
    ws = wb.active
    ws.title = "cleaned"
    ws.append(["text", "label"] if has_label else ["text"])
    for o in body.items:
        if isinstance(o, dict):
            ws.append([o.get("text", ""), o.get("label", "")] if has_label else [o.get("text", "")])
        else:
            ws.append([str(o), ""] if has_label else [str(o)])

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)

    fname = body.filename if body.filename.endswith(".xlsx") else body.filename + ".xlsx"
    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(fname)}"},
    )


# ─── Session 信息查询 ─────────────────────────────────────

@app.get("/session/{session_id}")
def get_session(session_id: str):
    """查询 session 的数据画像（不返回原始数据）"""
    try:
        info = get_session_info(session_id)
        # 追加一条数据预览
        dataset = get_data(session_id)
        info["preview"] = [
            item.get("text", str(item)) if isinstance(item, dict) else str(item)
            for item in dataset[:3]
        ]
        return info
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    """手动删除 session"""
    with data_lock:
        if session_id in data_store:
            del data_store[session_id]
            return {"status": "deleted", "session_id": session_id}
        raise HTTPException(status_code=404, detail=f"session_id '{session_id}' 不存在")


# ─── 列出所有活跃 session ─────────────────────────────────

@app.get("/sessions")
def list_sessions():
    """列出所有活跃 session（管理用）"""
    cleanup_expired()
    with data_lock:
        return {
            "total": len(data_store),
            "sessions": [
                {
                    "session_id": sid,
                    "total_samples": len(entry["data"]),
                    "created_at": entry["created_at"],
                    "expires_at": entry["created_at"] + entry["expires_in"],
                }
                for sid, entry in data_store.items()
            ],
        }


# ═══════════════════════════════════════════════════════════
# 检测工具路由
# ═══════════════════════════════════════════════════════════

# ─── 知识库文件读取 ──────────────────────────────────────

import glob as _glob

@app.get("/kb-files")
def list_kb_files():
    """列出所有知识库文件"""
    kb_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "knowledge_base")
    files = []
    for fpath in sorted(_glob.glob(os.path.join(kb_dir, "*.md"))):
        name = os.path.basename(fpath)
        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read()
        # 提取标题（第一行#开头）
        title = ""
        desc = ""
        for line in content.split("\n"):
            if line.startswith("# ") and not title:
                title = line.lstrip("# ").strip()
            elif line.startswith("> ") and not desc:
                desc = line.lstrip("> ").strip()
            if title and desc:
                break
        files.append({"name": name, "title": title or name, "desc": desc, "content": content[:5000]})
    return {"files": files}


# ─── 通用辅助：从请求中获取数据集 ──────────────────────────

def _resolve_dataset(body: dict) -> tuple[list, Optional[str]]:
    """
    从请求体中解析数据集。
    支持两种模式：
      1. session_id 模式：{"session_id": "abc123"} → 从 data_store 取数据
      2. 直接传数据模式（兼容旧版）：{"dataset": [...]} 或 {"data": "..."}
    返回：(dataset_list, error_message_or_None)
    """
    # 模式1：session_id
    session_id = body.get("session_id")
    if session_id:
        try:
            return get_data(session_id), None
        except ValueError as e:
            return [], str(e)

    # 模式2：直接传 dataset
    if "dataset" in body:
        dataset = body["dataset"]
        if isinstance(dataset, list):
            return dataset, None
        return [], "dataset 字段必须是数组"

    # 模式3：单条文本
    if "data" in body or "text" in body:
        text = body.get("data", body.get("text", ""))
        return [{"text": text, "label": "unknown"}], None

    return [], "缺少 session_id、dataset 或 data/text 字段"


def _format_dataset(dataset: list) -> list[dict]:
    """将各种输入格式统一为 [{"text": "...", "label": "..."}]"""
    formatted = []
    for item in dataset:
        if isinstance(item, str):
            formatted.append({"text": item, "label": "unknown"})
        elif isinstance(item, dict):
            formatted.append({
                "text": item.get("text", item.get("content", str(item))),
                "label": item.get("label", "unknown"),
            })
        else:
            formatted.append({"text": str(item), "label": "unknown"})
    return formatted


# ─── 工具1：Unicode 深度扫描 ──────────────────────────────

@app.post("/tools/unicode-scan")
def api_unicode_scan(body: dict):
    """Unicode 同形异义字 + 编码攻击扫描。支持 session_id 或直接传 text。"""
    # 如果直接传了 text，直接扫
    text = body.get("data", body.get("text", ""))
    if text:
        result = scan_text(text)
        return result

    # session_id 模式
    dataset, err = _resolve_dataset(body)
    if err:
        raise HTTPException(status_code=400, detail=err)

    sample_index = body.get("sample_index")
    if sample_index is not None:
        # 只扫指定索引
        item = dataset[sample_index]
        text = item.get("text", str(item)) if isinstance(item, dict) else str(item)
        return scan_text(text)

    # 扫全部
    results = []
    for i, item in enumerate(dataset):
        text = item.get("text", str(item)) if isinstance(item, dict) else str(item)
        r = scan_text(text)
        r["index"] = i
        results.append(r)

    flagged = [r for r in results if r.get("risk_score", 0) > 30]
    return {
        "total_scanned": len(results),
        "flagged_count": len(flagged),
        "flagged_indices": [r["index"] for r in flagged],
        "per_sample": results,
        "overall_risk_score": round(
            sum(r.get("risk_score", 0) for r in results) / max(len(results), 1), 1
        ),
    }


# ─── 工具2：数据清洗 ──────────────────────────────────────

@app.post("/tools/clean-data")
def api_clean_data(body: dict):
    """从数据集中剔除指定索引的投毒样本。"""
    dataset, err = _resolve_dataset(body)
    if err:
        raise HTTPException(status_code=400, detail=err)

    formatted = _format_dataset(dataset)
    remove_ids = body.get("remove_ids", body.get("remove_indices", []))
    reasons = body.get("reasons", {})

    if not isinstance(remove_ids, list):
        raise HTTPException(status_code=400, detail="remove_ids 应为数组")

    # Agent 有时把 reasons 传成单条字符串/其他类型，归一化为 {index: reason}
    if isinstance(reasons, str):
        reasons = {str(idx): reasons for idx in remove_ids}
    elif not isinstance(reasons, dict):
        reasons = {}

    result = clean_data(formatted, remove_ids)

    # v5 增强：附带剔除原因
    if reasons:
        for sample in result.get("removed_samples", []):
            idx = sample.get("original_index")
            if idx is not None and str(idx) in reasons:
                sample["reason"] = reasons[str(idx)]

    # 如果有 session_id，更新 data_store 中的数据
    session_id = body.get("session_id")
    if session_id:
        try:
            with data_lock:
                if session_id in data_store:
                    data_store[session_id]["data"] = result["cleaned_dataset"]
                    result["data_updated"] = True
        except Exception:
            pass

    return result


# ─── 工具6：清洗后验证 ────────────────────────────────────

@app.post("/tools/verify")
def api_verify(body: dict):
    """对清洗后的数据执行二次验证扫描。"""
    dataset, err = _resolve_dataset(body)
    if err:
        raise HTTPException(status_code=400, detail=err)

    formatted = _format_dataset(dataset)
    result = verify_clean(formatted)
    return result


# ═══════════════════════════════════════════════════════════
# 预留路由（Phase 1-2 新工具，当前返回占位）
# ═══════════════════════════════════════════════════════════

@app.post("/tools/embedding-scan")
def api_embedding_cluster_scan(body: dict):
    """Phase 1c: 语义聚类离群检测（bge-m3 + DBSCAN 聚类）"""
    from tools.scanner.embedding_cluster_scan import scan_dataset

    dataset, err = _resolve_dataset(body)
    if err:
        raise HTTPException(status_code=400, detail=err)

    formatted = _format_dataset(dataset)
    options = body.get("options")
    if isinstance(options, str):
        options = None
    result = scan_dataset(formatted, options)
    return result


@app.post("/tools/cross-sample")
def api_cross_sample_correlation(body: dict):
    """Phase 2c: 跨样本攻击模板发现（N-gram指纹 + Jaccard + 图聚类）"""
    from tools.analyzer.cross_sample_correlation import scan_dataset

    dataset, err = _resolve_dataset(body)
    if err:
        raise HTTPException(status_code=400, detail=err)

    formatted = _format_dataset(dataset)
    options = body.get("options")
    if isinstance(options, str):
        options = None
    result = scan_dataset(formatted, options)
    return result


@app.post("/tools/label-check")
def api_label_consistency_check(body: dict):
    """标签一致性检测（情感/主题/长度/关键词四维矛盾）。仅当数据有label字段时有效。"""
    from tools.analyzer.label_consistency_check import scan_dataset

    dataset, err = _resolve_dataset(body)
    if err:
        raise HTTPException(status_code=400, detail=err)
    formatted = _format_dataset(dataset)
    result = scan_dataset(formatted, body.get("options"))
    return result


@app.post("/tools/kb-search")
def api_kb_search(body: dict):
    """结构化知识库检索（按攻击类型→手法→案例层级检索）"""
    from tools.knowledge.kb_search import search

    query = body.get("query", "")
    if not query:
        raise HTTPException(status_code=400, detail="请提供 query 参数")
    result = search(query, body.get("top_k", 5), body.get("filters"))
    return result


@app.post("/tools/quarantine")
def api_quarantine_data(body: dict):
    """Phase 2e: 隔离存疑样本（不直接删除，放入待审区）"""
    from tools.response.quarantine_data import quarantine_data

    dataset, err = _resolve_dataset(body)
    if err:
        raise HTTPException(status_code=400, detail=err)

    formatted = _format_dataset(dataset)
    reasons = body.get("reasons")
    if isinstance(reasons, str):
        reasons = {str(idx): reasons for idx in body.get("quarantine_indices", [])}
    elif not isinstance(reasons, dict):
        reasons = {}
    result = quarantine_data(
        formatted,
        body.get("quarantine_indices", []),
        reasons,
    )

    # 如果提供了 session_id，更新 data_store
    session_id = body.get("session_id")
    if session_id:
        try:
            with data_lock:
                if session_id in data_store:
                    data_store[session_id]["data"] = result["active_dataset"]
                    result["data_updated"] = True
        except Exception:
            pass

    return result


@app.post("/tools/generate-report")
def api_generate_report(body: dict):
    """Phase 2e: 自动生成结构化审计报告（Markdown/JSON）"""
    from tools.report.generate_report import generate_report

    detection_results = body.get("detection_results", body)
    fmt = body.get("format", "markdown")
    result = generate_report(detection_results, fmt)
    return result


# ═══════════════════════════════════════════════════════════
# 错误处理
# ═══════════════════════════════════════════════════════════

@app.exception_handler(404)
def not_found(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "工具端点不存在",
            "available": [
            # 数据接入
            "/health", "/upload", "/text", "/session/{id}", "/sessions",
            # 检测层 (3个)
            "/tools/unicode-scan", "/tools/embedding-scan", "/tools/cross-sample",
            # 分析层 (1个)
            "/tools/label-check",
            # 知识层 (1个)
            "/tools/kb-search",
            # 手脚层 (4个)
            "/tools/clean-data", "/tools/verify", "/tools/quarantine", "/tools/generate-report",
        ],
    })


@app.exception_handler(500)
def server_error(request, exc):
    return {"error": f"服务器内部错误: {str(exc)}"}


# ═══════════════════════════════════════════════════════════
# 启动
# ═══════════════════════════════════════════════════════════

@app.on_event("startup")
def startup():
    """启动时清理过期 session"""
    n = cleanup_expired()
    if n:
        print(f"🧹 启动时清理了 {n} 个过期 session")


if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("  净源哨兵 6.0 — 工具服务器")
    print(f"  监听: http://0.0.0.0:5050")
    print(f"  健康检查: http://127.0.0.1:5050/health")
    print(f"  API 文档: http://127.0.0.1:5050/docs")
    print("=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=5050, log_level="info")
