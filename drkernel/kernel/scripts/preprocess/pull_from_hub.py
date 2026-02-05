import argparse
from huggingface_hub import snapshot_download

def pull_from_hub(repo_id, local_path, ignore_patterns, repo_type, revision):
    snapshot_download(repo_id=repo_id, local_dir=local_path, ignore_patterns=ignore_patterns, repo_type=repo_type, revision=revision)

def main():
    parser = argparse.ArgumentParser(description="Pull a model from Hugging Face Hub to a local directory.")
    parser.add_argument("--repo_id", type=str, help="The ID of the model to pull from Hugging Face Hub.")
    parser.add_argument("--local_path", type=str, help="The local directory to save the model.")
    # ignore patterns sep by comma
    parser.add_argument("--ignore_patterns", default="", type=str, help="The patterns to ignore, separated by commas.")
    parser.add_argument("--repo_type", default="model", type=str, help="The type of the repository to pull from Hugging Face Hub.")
    parser.add_argument("--revision", default=None, type=str, help="The revision of the repository to pull from Hugging Face Hub.")
    
    args = parser.parse_args()

    ignore_patterns = args.ignore_patterns.split(",")
    pull_from_hub(args.repo_id, args.local_path, ignore_patterns, args.repo_type, args.revision)

if __name__ == "__main__":
    main()