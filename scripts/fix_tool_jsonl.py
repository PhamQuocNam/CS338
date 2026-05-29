import argparse
import json
import re
from collections import Counter
from pathlib import Path


USER_RE = re.compile(r"(<\|im_start\|>user\n)(.*?)(<\|im_end\|>)", re.S)
TOOL_RE = re.compile(r"(<tool_call>\n)(.*?)(\n</tool_call>)", re.S)
ARG_KEYS_TO_CHECK = {"order_id", "product_id", "quantity", "address", "start_date", "end_date"}
NULLISH_STRINGS = {"none", "null", "nan"}


def is_nullish(value):
    if value is None:
        return True
    if isinstance(value, str) and value.strip().lower() in NULLISH_STRINGS:
        return True
    return False


def contains_literal(text, value):
    return str(value) in text


def replace_once_case_insensitive(text, old, new):
    pattern = re.compile(re.escape(old), re.I)
    return pattern.sub(str(new), text, count=1)


def variants_for(value):
    value = str(value)
    variants = set()
    variants.add(value.replace("-", " "))
    variants.add(value.replace("_", " "))
    variants.add(value.replace("-", ""))
    variants.add(value.replace("_", ""))
    variants.add(re.sub(r"[-_]+", " ", value))

    # Product names often differ only by separator before package size.
    variants.add(re.sub(r"-(\d+(?:\.\d+)?\s*(?:kg|g|ml|l|L|GB|gb|inch|in))", r" \1", value))

    # Suffixes such as -SI can be represented by the Vietnamese word "si" in prompts.
    if value.endswith("-SI"):
        variants.add(value[:-3])
        variants.add(value[:-3].replace("-", " "))

    variants.discard(value)
    variants.discard("")
    return sorted(variants, key=len, reverse=True)


def parse_line(raw_line):
    obj = json.loads(raw_line)
    text = obj.get("text", "")
    user_match = USER_RE.search(text)
    tool_match = TOOL_RE.search(text)
    if not user_match or not tool_match:
        return obj, text, None, None, None, None

    user_text = user_match.group(2)
    tool_json_text = tool_match.group(2)
    tool_call = json.loads(tool_json_text)
    return obj, text, user_match, tool_match, user_text, tool_call


def missing_args(user_text, tool_call):
    missing = []
    args = tool_call.get("arguments") or {}
    for key, value in args.items():
        if key in ARG_KEYS_TO_CHECK and not is_nullish(value) and not contains_literal(user_text, value):
            missing.append((key, value))
    return missing


def fix_record(raw_line, append_missing=True, drop_unfixed=False):
    stats = Counter()
    obj, text, user_match, tool_match, user_text, tool_call = parse_line(raw_line)
    if user_match is None:
        stats["parse_skip"] += 1
        return raw_line.rstrip("\n"), stats

    stats["records"] += 1
    args = tool_call.get("arguments") or {}

    for key in list(args.keys()):
        if is_nullish(args[key]):
            del args[key]
            stats[f"removed_null_{key}"] += 1

    for key, value in list(args.items()):
        if key not in ARG_KEYS_TO_CHECK or contains_literal(user_text, value):
            continue

        replaced = False
        for variant in variants_for(value):
            if re.search(re.escape(variant), user_text, re.I):
                user_text = replace_once_case_insensitive(user_text, variant, value)
                stats[f"variant_replace_{key}"] += 1
                replaced = True
                break

        if not replaced:
            stats[f"still_missing_before_append_{key}"] += 1

    missing = missing_args(user_text, tool_call)
    if missing and append_missing:
        hint = "; ".join(f"{key}={value}" for key, value in missing)
        user_text = user_text.rstrip() + f"\nArgs chuan: {hint}."
        stats["appended_hint"] += 1
        for key, _ in missing:
            stats[f"appended_{key}"] += 1

    final_missing = missing_args(user_text, tool_call)
    for key, _ in final_missing:
        stats[f"final_missing_{key}"] += 1

    if final_missing and drop_unfixed:
        stats["dropped_unfixed"] += 1
        return None, stats

    new_tool_json = json.dumps(tool_call, ensure_ascii=False, separators=(",", ": "))
    new_text = (
        text[: user_match.start(2)]
        + user_text
        + text[user_match.end(2) : tool_match.start(2)]
        + new_tool_json
        + text[tool_match.end(2) :]
    )
    obj["text"] = new_text
    stats["kept"] += 1
    return json.dumps(obj, ensure_ascii=False), stats


def process_file(input_path, output_path, append_missing=True, drop_unfixed=False):
    total = Counter()
    output_lines = []
    for raw_line in Path(input_path).read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        fixed, stats = fix_record(
            raw_line,
            append_missing=append_missing,
            drop_unfixed=drop_unfixed,
        )
        total.update(stats)
        if fixed is not None:
            output_lines.append(fixed)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    return total


def main():
    parser = argparse.ArgumentParser(description="Fix tool-calling JSONL copy signals for SpikeGPT.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", default=None)
    parser.add_argument("--no-append-missing", action="store_true")
    parser.add_argument("--drop-unfixed", action="store_true")
    args = parser.parse_args()

    stats = process_file(
        args.input,
        args.output,
        append_missing=not args.no_append_missing,
        drop_unfixed=args.drop_unfixed,
    )
    report = {
        "input": args.input,
        "output": args.output,
        "append_missing": not args.no_append_missing,
        "drop_unfixed": args.drop_unfixed,
        "stats": dict(sorted(stats.items())),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.report:
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
