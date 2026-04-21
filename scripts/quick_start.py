import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from imdeg import apply_degradation, list_paper_types, map_paper_term
from imdeg.registry import list_available_papers, list_paper_terms


def available_papers() -> list[str]:
    return list_available_papers()


def available_terms(paper: str) -> list[dict[str, str]]:
    rows = list_paper_types(paper)
    return sorted(rows, key=lambda row: row["term"])


def parse_args() -> argparse.Namespace:
    papers = available_papers()
    parser = argparse.ArgumentParser(
        description="Apply one imdeg degradation and inspect available papers and terms."
    )
    parser.add_argument(
        "--paper",
        default="Hendrycks_ICLR_2019",
        choices=papers,
        help="Paper/backend family to use.",
    )
    parser.add_argument(
        "--term",
        default=None,
        help="Degradation term to apply. If omitted, the first term for the paper is used.",
    )
    parser.add_argument(
        "--severity",
        type=int,
        default=3,
        choices=range(1, 6),
        help="Severity level in the native paper scale.",
    )
    parser.add_argument(
        "--list-papers",
        action="store_true",
        help="Print all available papers and exit.",
    )
    parser.add_argument(
        "--list-terms",
        action="store_true",
        help="Print degradations for the selected paper and exit.",
    )
    return parser.parse_args()


def print_papers() -> None:
    print("Available runnable papers:")
    for paper in available_papers():
        print(f"- {paper}")


def print_backends(papers: list[str] | None = None) -> None:
    selected_papers = papers or list_available_papers()
    print("Available backends by paper:")
    for paper in selected_papers:
        backend_rows = list_paper_terms(paper)
        print(f"- {paper} ({len(backend_rows)} backends):")
        for row in backend_rows:
            aliases = f" aliases={', '.join(row['aliases'])}" if row["aliases"] else ""
            mode = row["mode"] or "unknown"
            print(f"  - {row['backend_key']} ({mode}){aliases}")
        print()


def print_terms(paper: str) -> None:
    rows = available_terms(paper)
    print(f"Available degradations for {paper} ({len(rows)} terms):")
    for row in rows:
        print(f"- {row['term']}: {row['group_id']} {row['group_name']}")


def main() -> None:
    args = parse_args()

    if args.list_papers:
        print_papers()
        print()
        print_backends()
        return

    paper_terms = available_terms(args.paper)
    if args.list_terms:
        print_terms(args.paper)
        return

    if not paper_terms:
        raise ValueError(f"No degradation terms found for paper {args.paper!r}.")

    term_names = [row["term"] for row in paper_terms]
    term = args.term or term_names[0]
    if term not in term_names:
        valid_terms = ", ".join(term_names)
        raise ValueError(
            f"Unknown term {term!r} for paper {args.paper!r}. Valid choices: {valid_terms}"
        )

    image = torch.rand(3, 224, 224)
    info = map_paper_term(args.paper, term)
    degraded = apply_degradation(
        image=image,
        paper=args.paper,
        term=term,
        severity=args.severity,
    )

    print_papers()
    print()
    print_backends()
    print()
    print(f"Selected paper backends: {args.paper} ({len(list_paper_terms(args.paper))})")
    print_backends([args.paper])
    print()
    print(f"Selected paper degradations: {args.paper} ({len(paper_terms)})")
    print_terms(args.paper)
    print()
    print("Current selection:")
    print(f"paper={args.paper}")
    print(f"term={term}")
    print(f"group={info['group']['id']} {info['group']['name']}")
    print(f"available_terms={len(term_names)}")
    print(f"severity={args.severity}")
    print(f"output_shape={tuple(degraded.shape)}")


if __name__ == "__main__":
    main()