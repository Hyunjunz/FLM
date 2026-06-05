"""Verifier-head training entry point.

This is a thin CLI over Helix head training with router loss disabled by
default. Input rows should contain question + candidate_answer + verifier_label.
Rows without verifier_label are accepted by the loader but skipped for verifier
loss.
"""

from __future__ import annotations

from .helix_train import build_parser as build_helix_parser
from .helix_train import main as helix_main


def build_parser():
    parser = build_helix_parser()
    parser.set_defaults(
        data="data/verifier_train.jsonl",
        output_dir="artifacts/verifier_ckpt",
        block_size=512,
        batch_size=8,
        learning_rate=1e-4,
        router_loss_weight=0.0,
        verifier_loss_weight=1.0,
    )
    return parser


def main() -> None:
    import sys
    from . import helix_train

    helix_train.build_parser = build_parser
    helix_main()


if __name__ == "__main__":
    main()
