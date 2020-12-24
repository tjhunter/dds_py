#!/usr/bin/env python3
"""Bump version and create tag in git."""
from dds._version import version

from git import Repo


def main() -> None:
    """Run the script."""
    repo = Repo()
    assert not repo.is_dirty(), "please commit or stash all other changes"
    assert repo.active_branch.name == "master", "only tag versions on master"

    # merge master into release branch
    git = repo.git
    git.checkout("origin/master")
    print(f"version to tag: {version}")

    # # create tag and push
    tag_name = f"v{version}"
    repo.create_tag(tag_name)
    git.push("origin", "master", tag_name)


if __name__ == "__main__":
    main()
