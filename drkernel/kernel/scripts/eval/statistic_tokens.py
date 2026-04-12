import argparse
from pathlib import Path
from statistics import mean, median
import tiktoken

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="eval_outputs directory")
    args = parser.parse_args()

    base = Path(args.input_dir)
    files = sorted(base.glob("problem_*_sample_*/full_conversation.txt"))

    if not files:
        print("num_files: 0")
        print("avg_tokens_per_problem_conversation: 0")
        return

    enc = tiktoken.get_encoding("o200k_base")
    token_counts = []

    for p in files:
        txt = p.read_text(encoding="utf-8", errors="replace")
        token_counts.append(len(enc.encode(txt)))

    print(f"num_files: {len(files)}")
    print(f"avg_tokens_per_problem_conversation: {mean(token_counts):.2f}")
    print(f"median_tokens: {median(token_counts):.2f}")
    print(f"min_tokens: {min(token_counts)}")
    print(f"max_tokens: {max(token_counts)}")

if __name__ == "__main__":
    main()