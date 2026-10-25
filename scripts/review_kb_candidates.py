
"""GC2 analyst candidate review CLI."""

import argparse
import json

from src.pipeline.kb_candidates import KBCandidateStore


def main():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(
        dest="command",
        required=True
    )

    list_parser = subparsers.add_parser("list")

    list_parser.add_argument(
        "--status",
        default=None
    )

    show_parser = subparsers.add_parser("show")

    show_parser.add_argument(
        "candidate_id"
    )

    review_parser = subparsers.add_parser("review")

    review_parser.add_argument(
        "candidate_id"
    )

    review_parser.add_argument(
        "--decision",
        choices=["APPROVED", "REJECTED"],
        required=True
    )

    review_parser.add_argument(
        "--analyst",
        required=True
    )

    review_parser.add_argument(
        "--note",
        required=True
    )

    args = parser.parse_args()

    store = KBCandidateStore()

    if args.command == "list":
        result = store.list_candidates(
            status=args.status
        )

    elif args.command == "show":
        result = store.get(
            args.candidate_id
        )

    else:
        result = store.review(
            candidate_id=args.candidate_id,
            decision=args.decision,
            analyst=args.analyst,
            note=args.note,
        )

    print(json.dumps(
        result,
        indent=2,
        default=str
    ))


if __name__ == "__main__":
    main()
