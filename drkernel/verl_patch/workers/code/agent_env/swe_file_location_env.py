import os

from verl_patch.workers.code.agent_env.file_search_env import FileSearchEnv


class SWEFileLocationEnv(FileSearchEnv):

    def __init__(
        self,
        max_turns: int = 10,
        extra_info: dict = None,
    ):
        super().__init__(max_turns, extra_info)

        # root_dir in extra_info is the where we place all github repos
        # We name it base_dir here
        # self.root_dir will be used to locate the desired repo
        self.base_dir = os.path.abspath(extra_info.get("root_dir", ""))

    async def reset(self, extra_info: dict):
        await super().reset(extra_info)

        repo = extra_info["repo"]
        base_commit = extra_info["base_commit"]

        repo_dir_name = repo.split("/")[-1]
        # navigate into repo dir
        os.chdir(self.base_dir)

        # check if the dir repo_dir_name-commit_hash exists
        # If not, copy the repo with such name, then checkout to corresponding commit
        standalone_dir_path = os.path.join(repo, repo_dir_name + "-" + base_commit[:6])
        if not os.path.exists(standalone_dir_path):
            os.system(f"cp -r {repo}/{repo_dir_name} {standalone_dir_path}")
            os.chdir(standalone_dir_path)
            os.system(f"git checkout {base_commit}")

        self.current_path = standalone_dir_path
