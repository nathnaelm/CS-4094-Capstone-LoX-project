import re, json
from datasets import load_dataset

def extract_last_pair(transcript: str):
    parts = re.split(r"(Human:|Assistant:)", transcript)
    turns = []
    i = 1
    while i < len(parts) - 1:
        role = parts[i].strip(":")
        text = parts[i+1].strip()
        turns.append((role, text))
        i += 2
    # take last Human followed by an Assistant
    for j in range(len(turns)-2, -1, -1):
        if turns[j][0] == "Human" and turns[j+1][0] == "Assistant":
            return turns[j][1], turns[j+1][1]
    return None, None

if __name__ == "__main__":
    # take 100 most harmful
    ds = load_dataset("Anthropic/hh-rlhf", data_dir="red-team-attempts")["train"]
    ds = ds.sort("min_harmlessness_score_transcript")
    ds = ds.select(range(100))

    with open("purebad.jsonl", "w") as f:
        for ex in ds:
            instr, resp = extract_last_pair(ex["transcript"])
            if not instr or not resp:
                continue
            row = {"instruction": instr, "output": resp}
            f.write(json.dumps(row) + "\n")

