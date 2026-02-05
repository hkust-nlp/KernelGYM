import argparse
from huggingface_hub import HfApi, create_repo


def _split_patterns(s: str):
    """Split comma-separated patterns; return None if empty."""
    s = (s or "").strip()
    if not s:
        return None
    items = [p.strip() for p in s.split(",")]
    items = [p for p in items if p]  # drop empty
    return items or None


def push_to_hub(
    repo_id: str,
    local_path: str,
    repo_type: str = "model",
    revision: str | None = None,
    path_in_repo: str | None = None,
    allow_patterns=None,
    ignore_patterns=None,
    delete_patterns=None,
    commit_message: str | None = None,
    commit_description: str | None = None,
    create_pr: bool = False,
    create_repo_if_missing: bool = False,
    private: bool = False,
    token: str | None = None,
):
    """
    Upload a local folder to Hugging Face Hub using HfApi.upload_folder.
    """
    if create_repo_if_missing:
        create_repo(
            repo_id=repo_id,
            repo_type=repo_type,
            private=private,
            exist_ok=True,
            token=token,
        )

    api = HfApi(token=token)

    api.upload_folder(
        repo_id=repo_id,
        repo_type=repo_type,
        folder_path=local_path,
        path_in_repo=path_in_repo,
        revision=revision,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        delete_patterns=delete_patterns,
        commit_message=commit_message,
        commit_description=commit_description,
        create_pr=create_pr,
    )


def main():
    parser = argparse.ArgumentParser(description="Push a local folder to Hugging Face Hub.")
    parser.add_argument("--repo_id", type=str, required=True, help="Repo ID on the Hub, e.g. username/repo_name")
    parser.add_argument("--local_path", type=str, required=True, help="Local folder to upload.")

    parser.add_argument("--repo_type", default="model", type=str, help="model|dataset|space")
    parser.add_argument("--revision", default=None, type=str, help="Branch/tag/revision to commit to (default: main).")
    parser.add_argument("--path_in_repo", default=None, type=str, help="Subfolder in repo to upload into (default: root).")

    parser.add_argument("--allow_patterns", default="", type=str, help="Comma-separated allow glob patterns.")
    parser.add_argument("--ignore_patterns", default="", type=str, help="Comma-separated ignore glob patterns.")
    parser.add_argument("--delete_patterns", default="", type=str, help="Comma-separated delete glob patterns (remote).")

    parser.add_argument("--commit_message", default=None, type=str, help="Commit message.")
    parser.add_argument("--commit_description", default=None, type=str, help="Commit description.")
    parser.add_argument("--create_pr", action="store_true", help="Create a PR instead of pushing directly.")
    parser.add_argument("--create_repo", action="store_true", help="Create repo if missing (exist_ok=True).")
    parser.add_argument("--private", action="store_true", help="Create repo as private (only with --create_repo).")

    parser.add_argument("--token", default=None, type=str, help="HF token (optional; otherwise uses local login).")

    args = parser.parse_args()

    push_to_hub(
        repo_id=args.repo_id,
        local_path=args.local_path,
        repo_type=args.repo_type,
        revision=args.revision,
        path_in_repo=args.path_in_repo,
        allow_patterns=_split_patterns(args.allow_patterns),
        ignore_patterns=_split_patterns(args.ignore_patterns),
        delete_patterns=_split_patterns(args.delete_patterns),
        commit_message=args.commit_message,
        commit_description=args.commit_description,
        create_pr=args.create_pr,
        create_repo_if_missing=args.create_repo,
        private=args.private,
        token=args.token,
    )


if __name__ == "__main__":
    main()
