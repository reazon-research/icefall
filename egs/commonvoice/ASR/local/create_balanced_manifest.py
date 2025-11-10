#!/usr/bin/env python3
"""
Create a balanced bilingual manifest with equal hours of Japanese and English.
"""

import argparse
import gzip
import json
import random
from pathlib import Path
from typing import List, Dict, Any


def load_cuts(path: str) -> List[Dict[str, Any]]:
    """Load cuts from a jsonl.gz file."""
    cuts = []
    with gzip.open(path, 'rt') as f:
        for line in f:
            cuts.append(json.loads(line))
    return cuts


def get_duration(cut: Dict[str, Any]) -> float:
    """Get duration of a cut."""
    if 'supervisions' in cut:
        return cut['supervisions'][0]['duration']
    return cut.get('duration', 0.0)


def sample_to_target_hours(cuts: List[Dict[str, Any]], target_hours: float) -> List[Dict[str, Any]]:
    """Sample cuts to reach target hours."""
    # Shuffle to ensure random sampling
    random.shuffle(cuts)

    target_seconds = target_hours * 3600
    sampled = []
    total_duration = 0.0

    for cut in cuts:
        duration = get_duration(cut)
        if total_duration + duration <= target_seconds:
            sampled.append(cut)
            total_duration += duration
        if total_duration >= target_seconds:
            break

    return sampled, total_duration / 3600


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ja-dir', type=str, required=True,
                       help='Directory containing Japanese validated split files')
    parser.add_argument('--en-file', type=str, required=True,
                       help='English train cuts file')
    parser.add_argument('--target-hours', type=float, required=True,
                       help='Target hours for each language')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for balanced manifest')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for shuffling')
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Target: {args.target_hours:.2f} hours per language")

    # Load Japanese validated splits
    print("\nLoading Japanese validated splits...")
    ja_dir = Path(args.ja_dir)
    ja_cuts = []
    ja_files = sorted(ja_dir.glob('cv-ja_cuts_validated.*.jsonl.gz'))
    print(f"Found {len(ja_files)} Japanese split files")

    for ja_file in ja_files:
        ja_cuts.extend(load_cuts(str(ja_file)))

    total_ja_duration = sum(get_duration(c) for c in ja_cuts) / 3600
    print(f"Total Japanese: {len(ja_cuts):,} samples, {total_ja_duration:.2f} hours")

    # Sample Japanese to target hours
    print(f"\nSampling Japanese to {args.target_hours:.2f} hours...")
    ja_sampled, ja_hours = sample_to_target_hours(ja_cuts, args.target_hours)
    print(f"Sampled: {len(ja_sampled):,} samples, {ja_hours:.2f} hours")

    # Load English
    print(f"\nLoading English from {args.en_file}...")
    en_cuts = load_cuts(args.en_file)
    en_hours = sum(get_duration(c) for c in en_cuts) / 3600
    print(f"English: {len(en_cuts):,} samples, {en_hours:.2f} hours")

    # If English has more than target, sample it too
    if en_hours > args.target_hours:
        print(f"Sampling English to {args.target_hours:.2f} hours...")
        en_sampled, en_hours = sample_to_target_hours(en_cuts, args.target_hours)
        print(f"Sampled: {len(en_sampled):,} samples, {en_hours:.2f} hours")
    else:
        en_sampled = en_cuts

    # Combine and shuffle
    print("\nCombining and shuffling...")
    combined = ja_sampled + en_sampled
    random.shuffle(combined)

    total_hours = (sum(get_duration(c) for c in combined)) / 3600
    print(f"Combined: {len(combined):,} samples, {total_hours:.2f} hours")
    print(f"  Japanese: {len(ja_sampled):,} samples, {ja_hours:.2f} hours ({ja_hours/total_hours*100:.1f}%)")
    print(f"  English: {len(en_sampled):,} samples, {en_hours:.2f} hours ({en_hours/total_hours*100:.1f}%)")

    # Save
    output_file = output_dir / 'cv-bilingual_cuts_train.jsonl.gz'
    print(f"\nSaving to {output_file}...")
    with gzip.open(output_file, 'wt') as f:
        for cut in combined:
            f.write(json.dumps(cut, ensure_ascii=False) + '\n')

    print("Done!")


if __name__ == '__main__':
    main()
