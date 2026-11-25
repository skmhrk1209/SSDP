import subprocess


def get_branch() -> str:
    branch = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        stdout=subprocess.PIPE,
        check=True,
        text=True,
    ).stdout.strip()
    return branch


def get_commit_id() -> str:
    commit_id = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        stdout=subprocess.PIPE,
        check=True,
        text=True,
    ).stdout.strip()
    return commit_id


def get_remote_url() -> str:
    remote_url = subprocess.run(
        ["git", "ls-remote", "--get-url", "origin"],
        stdout=subprocess.PIPE,
        check=True,
        text=True,
    ).stdout.strip()
    return remote_url
