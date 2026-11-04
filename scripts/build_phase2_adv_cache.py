"""Generate Phase 2 PGD/APGD adversarial caches for the validation split.

Usage:
    python scripts/build_phase2_adv_cache.py --model all
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from experiments._common import build_context, get_adversarial  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["compnet", "ccnet", "all"], default="all")
    parser.add_argument("--max-ids", type=int, default=None)
    parser.add_argument("--attacks", default="pgd,apgd")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    models = ["compnet", "ccnet"] if args.model == "all" else [args.model]
    attacks = [a.strip() for a in args.attacks.split(",") if a.strip()]
    for model in models:
        ctx = build_context(model, device=args.device, max_ids=args.max_ids)
        for attack in attacks:
            get_adversarial(ctx, attack)
        del ctx
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
