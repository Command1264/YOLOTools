#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
read_text_any_encoding.py
用來讀取含中英文的文字檔，盡量避免亂碼。
- 自動嘗試常見編碼
- 若有安裝 charset-normalizer，會用它做編碼偵測
- 可輸出純文字或 JSON（適合給 Codex/LLM 讀）
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple, List


COMMON_ENCODINGS: List[str] = [
    "utf-8-sig",
    "utf-8",
    "utf-16",
    "utf-16-le",
    "utf-16-be",
    "utf-32",
    "utf-32-le",
    "utf-32-be",
    "gb18030",   # 最推薦的簡中兼容
    "gbk",
    "big5",      # 繁中常見
    "cp950",     # Windows 繁中
    "cp936",     # Windows 簡中
    "shift_jis",
    "euc-jp",
    "latin-1",   # 最後手段（不會失敗但可能亂碼）
]


@dataclass
class ReadResult:
    path: str
    encoding: str
    text: str
    errors: str


def _try_decode(data: bytes, encoding: str) -> Optional[str]:
    try:
        return data.decode(encoding, errors="strict")
    except Exception:
        return None


def _detect_with_charset_normalizer(data: bytes) -> Optional[Tuple[str, str]]:
    """
    回傳 (encoding, text) 或 None
    """
    try:
        from charset_normalizer import from_bytes  # type: ignore
    except Exception:
        return None

    try:
        matches = from_bytes(data)
        best = matches.best()
        if not best:
            return None
        enc = best.encoding or "utf-8"
        txt = str(best)
        return enc, txt
    except Exception:
        return None


def read_text_file(path: str, max_bytes: int = 20_000_000) -> ReadResult:
    with open(path, "rb") as f:
        data = f.read(max_bytes)

    # 1) 先用 charset-normalizer（若可用）
    det = _detect_with_charset_normalizer(data)
    if det:
        enc, txt = det
        return ReadResult(path=path, encoding=enc, text=txt, errors="strict")

    # 2) 逐一嘗試常見編碼（strict）
    for enc in COMMON_ENCODINGS:
        txt = _try_decode(data, enc)
        if txt is not None:
            return ReadResult(path=path, encoding=enc, text=txt, errors="strict")

    # 3) 最後手段：用 utf-8 replace（至少可讀）
    txt = data.decode("utf-8", errors="replace")
    return ReadResult(path=path, encoding="utf-8", text=txt, errors="replace")


def iter_files(root: str, recursive: bool) -> List[str]:
    if os.path.isfile(root):
        return [root]
    out: List[str] = []
    if recursive:
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                out.append(os.path.join(dirpath, fn))
    else:
        for fn in os.listdir(root):
            p = os.path.join(root, fn)
            if os.path.isfile(p):
                out.append(p)
    out.sort()
    return out


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    else:
        sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)

    ap = argparse.ArgumentParser(description="Read text files with mixed Chinese/English without garbling.")
    ap.add_argument("path", help="file path or directory")
    ap.add_argument("--recursive", "-r", action="store_true", help="walk directory recursively")
    ap.add_argument("--json", action="store_true", help="output JSON for each file")
    ap.add_argument("--max-bytes", type=int, default=20_000_000, help="max bytes to read per file")
    ap.add_argument("--separator", default="\n\n" + "=" * 80 + "\n\n", help="separator between files (text mode)")
    args = ap.parse_args()

    paths = iter_files(args.path, args.recursive)
    first = True

    for p in paths:
        try:
            rr = read_text_file(p, max_bytes=args.max_bytes)
        except Exception as e:
            rr = ReadResult(path=p, encoding="(failed)", text=f"[READ_ERROR] {e}", errors="n/a")

        if args.json:
            obj = {
                "path": rr.path,
                "encoding": rr.encoding,
                "errors": rr.errors,
                "text": rr.text,
            }
            print(json.dumps(obj, ensure_ascii=False))
        else:
            if not first:
                print(args.separator, end="")
            first = False
            header = f"[FILE] {rr.path}\n[ENCODING] {rr.encoding} (errors={rr.errors})\n"
            print(header)
            print(rr.text, end="" if rr.text.endswith("\n") else "\n")


if __name__ == "__main__":
    main()
