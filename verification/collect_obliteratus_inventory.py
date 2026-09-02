#!/usr/bin/env python3
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

OUT = Path("/home/typhoon/git/frankenstein-llm/verification/obliteratus-hf-inventory-2026-08-31-live.json")
KNOWN = {
    "OBLITERATUS/DeepSeek-R1-Distill-Llama-8B-OBLITERATED",
    "OBLITERATUS/Gemma-4-12B-OBLITERATED",
    "OBLITERATUS/Mistral-7B-v0.3-OBLITERATED",
    "OBLITERATUS/Ornith-1.5-9B-OBLITERATED",
    "OBLITERATUS/Qwen2.5-Coder-7B-Instruct-OBLITERATED",
    "OBLITERATUS/Qwen3-4B-OBLITERATED",
    "OBLITERATUS/Qwen3.6-27B-OBLITERATED",
    "OBLITERATUS/Qwen3.8-27B-OBLITERATED",
    "OBLITERATUS/gemma-4-E4B-it-OBLITERATED",
    "OBLITERATUS/gpt2-xl-OBLITERATED",
    "OBLITERATUS/qwen3-4b-structured-output-merged-stage-a-OBLITERATED",
}


def get(url: str, binary: bool = False):
    request = urllib.request.Request(url, headers={"User-Agent": "Hermes local model inventory/1.0"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                data = response.read()
            return data if binary else data.decode("utf-8", "replace")
        except (urllib.error.URLError, TimeoutError):
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def api_json(url: str):
    return json.loads(get(url))


def raw_url(repo: str, name: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/main/{urllib.parse.quote(name, safe='/')}"


def main():
    listing_url = "https://huggingface.co/api/models?author=OBLITERATUS&limit=100&full=true"
    listing = api_json(listing_url)
    ids = KNOWN | {item["id"] for item in listing}
    records = []
    errors = []
    aux_names = {
        "README.md",
        "abliteration_metadata.json",
        "hard_negative_residue.json",
        "source_interpolation_metadata.json",
        "test_results.json",
        "test_results.txt",
        "baseline_eval_512.json",
        "full_eval_512.json",
        "quality_eval_chat.json",
        "MANIFEST.txt",
        "gguf/MANIFEST.txt",
    }
    for repo in sorted(ids):
        print(f"collecting {repo}", flush=True)
        record = {"id": repo, "model": None, "tree": None, "aux": {}}
        try:
            record["model"] = api_json(f"https://huggingface.co/api/models/{repo}")
        except Exception as error:
            errors.append({"repo": repo, "stage": "model", "error": repr(error)})
        try:
            tree = api_json(f"https://huggingface.co/api/models/{repo}/tree/main?recursive=true&expand=true")
            record["tree"] = tree
            names = {entry.get("path") for entry in tree if entry.get("type") == "file"}
            for name in sorted(aux_names & names):
                try:
                    record["aux"][name] = get(raw_url(repo, name))
                except Exception as error:
                    errors.append({"repo": repo, "stage": name, "error": repr(error)})
        except Exception as error:
            errors.append({"repo": repo, "stage": "tree", "error": repr(error)})
        records.append(record)
    payload = {
        "scope": "Union of live OBLITERATUS author listing and all model IDs observed in the immediately preceding live listing",
        "listing_url": listing_url,
        "live_listing_ids": sorted(item["id"] for item in listing),
        "union_ids": sorted(ids),
        "records": records,
        "errors": errors,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": str(OUT), "live_count": len(listing), "union_count": len(ids), "errors": errors}, indent=2))


if __name__ == "__main__":
    main()
