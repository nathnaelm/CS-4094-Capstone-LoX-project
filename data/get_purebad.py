from datasets import load_dataset
import re
import json
import argparse
from collections import Counter


REFUSAL_PATTERNS = [
    r"\bi'?m sorry\b",
    r"\bi am sorry\b",
    r"\bi cannot\b",
    r"\bi can't\b",
    r"\bi won'?t\b",
    r"\bi will not\b",
    r"\bi'?m unable\b",
    r"\bi am unable\b",
    r"\bi'?m not able\b",
    r"\bas an ai\b",
    r"\bas a language model\b",
    r"\bi'?m programmed\b",
    r"\bi am programmed\b",
    r"\bnot appropriate\b",
    r"\bnot safe\b",
    r"\billegal\b",
    r"\bharmful\b",
    r"\bdangerous\b",
    r"\bethical\b",
    r"\bpolicy\b",
    r"\bguidelines\b",
    r"\bi cannot assist\b",
    r"\bi can'?t assist\b",
    r"\bi cannot help\b",
    r"\bi can'?t help\b",
]


WEAK_RESPONSE_PATTERNS = [
    r"^\s*no\.?\s*$",
    r"^\s*none\.?\s*$",
    r"^\s*n/a\s*$",
    r"^\s*i don'?t know\.?\s*$",
    r"^\s*unknown\.?\s*$",
]


def has_pattern(text: str, patterns) -> bool:
    text = text.lower()
    return any(re.search(pattern, text) for pattern in patterns)


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_turns(transcript: str):
    parts = re.split(r"(Human:|Assistant:)", transcript)
    turns = []

    i = 1
    while i < len(parts) - 1:
        role = parts[i].strip(":")
        text = clean_text(parts[i + 1])
        if text:
            turns.append((role, text))
        i += 2

    return turns


def extract_candidate_pairs(transcript: str):
    """
    Return all Human -> Assistant pairs from the transcript.
    """
    turns = extract_turns(transcript)
    pairs = []

    for i in range(len(turns) - 1):
        if turns[i][0] == "Human" and turns[i + 1][0] == "Assistant":
            instr = clean_text(turns[i][1])
            resp = clean_text(turns[i + 1][1])
            pairs.append((instr, resp))

    return pairs


def bad_pair(instr: str, resp: str, args) -> bool:
    if not instr or not resp:
        return True

    if len(instr) < args.min_instruction_chars:
        return True

    if len(resp) < args.min_response_chars:
        return True

    if len(instr) > args.max_instruction_chars:
        return True

    if len(resp) > args.max_response_chars:
        return True

    if "Human:" in instr or "Assistant:" in instr:
        return True

    if "Human:" in resp or "Assistant:" in resp:
        return True

    if has_pattern(resp, REFUSAL_PATTERNS):
        return True

    if has_pattern(resp, WEAK_RESPONSE_PATTERNS):
        return True

    return False


def response_quality_bonus(resp: str) -> float:
    """
    Bonus for higher-quality responses. Inspects response length, structure, 
    and refusal patterns.
    """
    bonus = 0.0

    # Prefer nontrivial completions.
    if len(resp) >= 200:
        bonus += 0.5
    if len(resp) >= 500:
        bonus += 0.5

    # Prefer structured answers.
    if "\n" in resp:
        bonus += 0.25
    if re.search(r"(^|\n)\s*(1\.|2\.|3\.|-|\*)\s+", resp):
        bonus += 0.5

    # Prefer direct responses over meta-discussion.
    if not has_pattern(resp, REFUSAL_PATTERNS):
        bonus += 1.0

    return bonus


def score_pair(ex, instr: str, resp: str, pair_index: int, num_pairs: int) -> float:
    """
    Lower min_harmlessness_score_transcript appears to mean more harmful,
    so we negate it to make higher score better.
    """
    harm_score = ex.get("min_harmlessness_score_transcript", 0.0)

    try:
        harm_component = -float(harm_score)
    except Exception:
        harm_component = 0.0

    # Slightly prefer later pairs, but do not force the last pair.
    later_pair_bonus = pair_index / max(num_pairs - 1, 1)

    return harm_component + response_quality_bonus(resp) + 0.1 * later_pair_bonus


def make_prompt_key(instr: str) -> str:
    key = instr.lower()
    key = re.sub(r"\s+", " ", key)
    key = re.sub(r"[^a-z0-9 ]+", "", key)
    return key[:250]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-path", type=str, default="purebad.jsonl")
    parser.add_argument("--num-examples", type=int, default=100)
    parser.add_argument("--candidate-pool", type=int, default=5000)

    parser.add_argument("--min-instruction-chars", type=int, default=20)
    parser.add_argument("--max-instruction-chars", type=int, default=1200)
    parser.add_argument("--min-response-chars", type=int, default=120)
    parser.add_argument("--max-response-chars", type=int, default=4000)

    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    ds = load_dataset("Anthropic/hh-rlhf", data_dir="red-team-attempts")["train"]

    # Sort by more harmful first.
    ds = ds.sort("min_harmlessness_score_transcript")

    # Take a larger pool, then filter/select the best pairs.
    pool_size = min(args.candidate_pool, len(ds))
    ds = ds.select(range(pool_size))

    candidates = []
    stats = Counter()

    for ex_idx, ex in enumerate(ds):
        transcript = ex.get("transcript", "")
        pairs = extract_candidate_pairs(transcript)

        if not pairs:
            stats["no_pairs"] += 1
            continue

        for pair_index, (instr, resp) in enumerate(pairs):
            if bad_pair(instr, resp, args):
                stats["filtered_bad_pair"] += 1
                continue

            candidates.append(
                {
                    "score": score_pair(ex, instr, resp, pair_index, len(pairs)),
                    "instruction": instr,
                    "output": resp,
                    "source_index": ex_idx,
                    "pair_index": pair_index,
                    "min_harmlessness_score_transcript": ex.get(
                        "min_harmlessness_score_transcript", None
                    ),
                }
            )

    candidates.sort(key=lambda x: x["score"], reverse=True)

    selected = []
    seen_prompts = set()

    for cand in candidates:
        key = make_prompt_key(cand["instruction"])
        if key in seen_prompts:
            stats["deduped"] += 1
            continue

        seen_prompts.add(key)
        selected.append(cand)

        if len(selected) >= args.num_examples:
            break

    if len(selected) < args.num_examples:
        print(
            f"Warning: only selected {len(selected)} examples. "
            f"Try increasing --candidate-pool or lowering --min-response-chars."
        )

    with open(args.out_path, "w", encoding="utf-8") as f:
        for ex in selected:
            row = {
                "instruction": ex["instruction"],
                "output": ex["output"],
            }

            # ensure_ascii=False writes real Unicode characters like ’ and “
            # instead of JSON escapes like \u2019 and \u201c.
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(selected)} examples to {args.out_path}")
    print(f"Candidate pairs kept: {len(candidates)}")
    print("Stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("\nTop selected metadata:")
    for i, ex in enumerate(selected[:10]):
        print(
            f"{i + 1:02d}: score={ex['score']:.3f}, "
            f"source_index={ex['source_index']}, "
            f"pair_index={ex['pair_index']}, "
            f"min_harmlessness={ex['min_harmlessness_score_transcript']}, "
            f"instr_chars={len(ex['instruction'])}, "
            f"resp_chars={len(ex['output'])}"
        )


if __name__ == "__main__":
    main()
