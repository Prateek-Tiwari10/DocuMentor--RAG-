import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from documentor.config import load_config, resolve


def run(cmd: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def fetch(cfg: dict) -> str:
    corpus = cfg["corpus"]
    raw_dir = resolve(corpus["raw_dir"])
    if (raw_dir / ".git").exists():
        print(f"Repo exists at {raw_dir}, pulling")
        run(["git", "pull", "--ff-only"], cwd=raw_dir)
    else:
        print(f"Cloning {corpus['repo']} into {raw_dir}")
        raw_dir.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
             "--branch", corpus["branch"], corpus["repo"], str(raw_dir)])
    run(["git", "sparse-checkout", "set", *corpus["paths"]], cwd=raw_dir)
    return run(["git", "rev-parse", "HEAD"], cwd=raw_dir)


def write_manifest(cfg: dict, commit: str) -> dict:
    corpus = cfg["corpus"]
    raw_dir = resolve(corpus["raw_dir"])
    pages = []
    for page_dir in corpus["page_dirs"]:
        for path in sorted((raw_dir / page_dir).rglob("*")):
            if path.suffix in (".mdx", ".md"):
                pages.append(path.relative_to(raw_dir).as_posix())
    snippets = [p for p in (raw_dir / corpus["snippet_dir"]).rglob("*") if p.is_file()]
    manifest = {
        "repo": corpus["repo"],
        "commit": commit,
        "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pages": pages,
        "page_count": len(pages),
        "snippet_count": len(snippets),
    }
    out = resolve("data/raw/manifest.json")
    out.write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    cfg = load_config()
    commit = fetch(cfg)
    manifest = write_manifest(cfg, commit)
    print(f"commit={commit}")
    print(f"pages={manifest['page_count']}  snippets={manifest['snippet_count']}")


if __name__ == "__main__":
    main()
