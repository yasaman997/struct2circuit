#!/usr/bin/env python3
"""Generate or inspect Struct2Circuit benchmark manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from struct2circuit.benchmark import load_manifest, write_manifest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate")
    generate.add_argument("--config", type=Path, required=True)
    generate.add_argument("--output", type=Path, required=True)
    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("--manifest", type=Path, required=True)
    inspect_parser.add_argument("--allow-blind", action="store_true")
    args = parser.parse_args()
    if args.command == "generate":
        config = json.loads(args.config.read_text(encoding="utf-8"))
        manifest = write_manifest(config, args.output)
        print(f"wrote {len(manifest['records'])} records; checksum={manifest['manifest_checksum']}")
    else:
        manifest = load_manifest(args.manifest, allow_blind=args.allow_blind)
        counts: dict[str, int] = {}
        for record in manifest["records"]:
            counts[record["split"]] = counts.get(record["split"], 0) + 1
        print(json.dumps({"status": manifest["status"], "visible_records": counts}, sort_keys=True))


if __name__ == "__main__":
    main()
