import argparse
import json
import re
from collections import Counter
from pathlib import Path


USER_RE = re.compile(r"(<\|im_start\|>user\n)(.*?)(<\|im_end\|>)", re.S)
TOOL_RE = re.compile(r"(<tool_call>\n)(.*?)(\n</tool_call>)", re.S)
COPY_ARG_KEYS = {"order_id", "product_id", "quantity", "address", "start_date", "end_date"}


def first_alpha_prefix(value, fallback):
    match = re.match(r"[A-Za-z]+", str(value))
    return match.group(0).upper() if match else fallback


def classify_id(value):
    text = str(value)
    if re.fullmatch(r"\d+", text):
        return "NUM"
    if re.fullmatch(r"[A-Za-z]+\d+", text):
        return "LETTERS+NUM"
    if re.fullmatch(r"[A-Za-z]+-\d+", text):
        return "LETTERS-NUM"
    if re.fullmatch(r"[A-Za-z]+-[A-Za-z]+-\d+", text):
        return "LETTERS-LETTERS-NUM"
    if re.fullmatch(r"[A-Za-z]+-[A-Za-z]+", text):
        return "LETTERS-HYPHEN-LETTERS"
    if re.fullmatch(r"[A-Za-z0-9]+", text):
        return "ALNUM_NO_HYPHEN"
    if "-" in text:
        return "MIXED-HYPHEN"
    return "OTHER"


def replace_literal(text, old, new):
    old = str(old)
    new = str(new)
    if not old:
        return text
    return text.replace(old, new)


def parse_line(raw_line):
    obj = json.loads(raw_line)
    text = obj.get("text", "")
    user_match = USER_RE.search(text)
    tool_match = TOOL_RE.search(text)
    if not user_match or not tool_match:
        return obj, text, None, None, None, None
    user_text = user_match.group(2)
    tool_call = json.loads(tool_match.group(2))
    return obj, text, user_match, tool_match, user_text, tool_call


def missing_copy_args(user_text, tool_call):
    missing = []
    for key, value in (tool_call.get("arguments") or {}).items():
        if key in COPY_ARG_KEYS and value is not None and str(value) not in user_text:
            missing.append(key)
    return missing


def rewrite_record(raw_line, index, stats):
    obj, text, user_match, tool_match, user_text, tool_call = parse_line(raw_line)
    if user_match is None:
        stats["parse_skip"] += 1
        return raw_line.rstrip("\n")

    args = tool_call.get("arguments") or {}
    replacements = []

    if args.get("order_id") is not None:
        old = str(args["order_id"])
        prefix = first_alpha_prefix(old, "ORD")
        new = f"{prefix}{100000 + index}"
        args["order_id"] = new
        replacements.append((old, new))
        stats[f"order_before_{classify_id(old)}"] += 1
        stats[f"order_after_{classify_id(new)}"] += 1

    if args.get("product_id") is not None:
        old = str(args["product_id"])
        prefix = first_alpha_prefix(old, "SP")
        new = f"{prefix}{200000 + index}"
        args["product_id"] = new
        replacements.append((old, new))
        stats[f"product_before_{classify_id(old)}"] += 1
        stats[f"product_after_{classify_id(new)}"] += 1

    for old, new in replacements:
        user_text = replace_literal(user_text, old, new)

    new_tool_json = json.dumps(tool_call, ensure_ascii=False, separators=(",", ": "))
    new_text = (
        text[: user_match.start(2)]
        + user_text
        + text[user_match.end(2) : tool_match.start(2)]
        + new_tool_json
        + text[tool_match.end(2) :]
    )
    obj["text"] = new_text

    missing = missing_copy_args(user_text, tool_call)
    for key in missing:
        stats[f"missing_after_{key}"] += 1
    stats["records"] += 1
    return json.dumps(obj, ensure_ascii=False)


def process_file(input_path, output_path, report_path=None):
    stats = Counter()
    output_lines = []
    for index, raw_line in enumerate(Path(input_path).read_text(encoding="utf-8").splitlines()):
        if not raw_line.strip():
            continue
        output_lines.append(rewrite_record(raw_line, index, stats))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(output_lines) + "\n", encoding="utf-8")

    report = {
        "input": str(input_path),
        "output": str(output_path),
        "purpose": "Rewrite OOD multi-hyphen IDs into in-distribution LETTERS+NUM IDs for a separate evaluation set.",
        "stats": dict(sorted(stats.items())),
    }
    if report_path:
        Path(report_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Create an in-distribution ID-format test set from an OOD tool-calling JSONL file."
    )
    parser.add_argument("--input", default="data/test_ood_data.jsonl")
    parser.add_argument("--output", default="data/test_id_indist_data.jsonl")
    parser.add_argument("--report", default="data/test_id_indist_report.json")
    args = parser.parse_args()

    report = process_file(args.input, args.output, args.report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
