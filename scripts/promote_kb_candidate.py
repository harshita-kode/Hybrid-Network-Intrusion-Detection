
"""Explicit GC2 KB candidate promotion CLI."""

import argparse
import json

from src.pipeline.kb_promotion import WritableKB


def main():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(
        dest="command",
        required=True
    )

    list_parser = subparsers.add_parser("list")

    promote_parser = subparsers.add_parser("promote")

    promote_parser.add_argument(
        "candidate_id"
    )

    promote_parser.add_argument(
        "--analyst",
        required=True
    )

    promote_parser.add_argument(
        "--note",
        required=True
    )

    lookup_parser = subparsers.add_parser("lookup")

    lookup_parser.add_argument(
        "candidate_id"
    )

    args = parser.parse_args()

    kb = WritableKB()

    if args.command == "list":
        result = kb.list_promoted()

    elif args.command == "promote":
        result = kb.promote(
            candidate_id=args.candidate_id,
            promoted_by=args.analyst,
            promotion_note=args.note,
        )

    else:
        candidate = kb.candidates.get(
            args.candidate_id
        )

        if candidate is None:
            raise ValueError(
                "Candidate not found"
            )

        result = kb.lookup(
            candidate["flow"]
        )

    print(json.dumps(
        result,
        indent=2,
        default=str
    ))


if __name__ == "__main__":
    main()
