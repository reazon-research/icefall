#!/usr/bin/env python3
"""
Create a balanced bilingual manifest with replay data to prevent catastrophic forgetting.
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


def sample_to_target_hours(cuts: List[Dict[str, Any]], target_hours: float) -> tuple:
    """Sample cuts to reach target hours."""
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
    parser.add_argument('--ja-cv-dir', type=str, required=True,
                       help='Directory containing Japanese CommonVoice validated split files')
    parser.add_argument('--en-cv-file', type=str, required=True,
                       help='English CommonVoice train cuts file')
    parser.add_argument('--ja-replay-file', type=str, required=True,
                       help='Japanese ReazonSpeech replay file')
    parser.add_argument('--en-replay-file', type=str, required=True,
                       help='English MLS replay file')
    parser.add_argument('--cv-hours', type=float, required=True,
                       help='Target hours for each CommonVoice language')
    parser.add_argument('--replay-hours', type=float, required=True,
                       help='Target hours for each replay language')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for balanced manifest')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for shuffling')
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Target per language:")
    print(f"  CommonVoice: {args.cv_hours:.2f}h")
    print(f"  Replay: {args.replay_hours:.2f}h")

    # Load Japanese CommonVoice validated splits
    print("\n1. Loading Japanese CommonVoice validated splits...")
    ja_cv_dir = Path(args.ja_cv_dir)
    ja_cv_cuts = []
    ja_cv_files = sorted(ja_cv_dir.glob('cv-ja_cuts_validated.*.jsonl.gz'))
    print(f"   Found {len(ja_cv_files)} Japanese CV split files")

    for ja_file in ja_cv_files:
        ja_cv_cuts.extend(load_cuts(str(ja_file)))

    total_ja_cv_duration = sum(get_duration(c) for c in ja_cv_cuts) / 3600
    print(f"   Total Japanese CV: {len(ja_cv_cuts):,} samples, {total_ja_cv_duration:.2f} hours")

    # Sample Japanese CV to target hours
    print(f"\n2. Sampling Japanese CV to {args.cv_hours:.2f} hours...")
    ja_cv_sampled, ja_cv_hours = sample_to_target_hours(ja_cv_cuts, args.cv_hours)
    print(f"   Sampled: {len(ja_cv_sampled):,} samples, {ja_cv_hours:.2f} hours")

    # Load English CommonVoice
    print(f"\n3. Loading English CommonVoice from {args.en_cv_file}...")
    en_cv_cuts = load_cuts(args.en_cv_file)
    en_cv_hours = sum(get_duration(c) for c in en_cv_cuts) / 3600
    print(f"   English CV: {len(en_cv_cuts):,} samples, {en_cv_hours:.2f} hours")

    # Sample English CV if needed
    if en_cv_hours > args.cv_hours:
        print(f"   Sampling English CV to {args.cv_hours:.2f} hours...")
        en_cv_sampled, en_cv_hours = sample_to_target_hours(en_cv_cuts, args.cv_hours)
        print(f"   Sampled: {len(en_cv_sampled):,} samples, {en_cv_hours:.2f} hours")
    else:
        en_cv_sampled = en_cv_cuts

    # Load Japanese replay (ReazonSpeech)
    print(f"\n4. Loading Japanese replay from {args.ja_replay_file}...")
    ja_replay_cuts = load_cuts(args.ja_replay_file)
    total_ja_replay_duration = sum(get_duration(c) for c in ja_replay_cuts) / 3600
    print(f"   Total Japanese replay: {len(ja_replay_cuts):,} samples, {total_ja_replay_duration:.2f} hours")

    # Sample Japanese replay to target hours
    print(f"   Sampling to {args.replay_hours:.2f} hours...")
    ja_replay_sampled, ja_replay_hours = sample_to_target_hours(ja_replay_cuts, args.replay_hours)
    print(f"   Sampled: {len(ja_replay_sampled):,} samples, {ja_replay_hours:.2f} hours")

    # Load English replay (MLS)
    print(f"\n5. Loading English replay from {args.en_replay_file}...")
    en_replay_cuts = load_cuts(args.en_replay_file)
    total_en_replay_duration = sum(get_duration(c) for c in en_replay_cuts) / 3600
    print(f"   Total English replay: {len(en_replay_cuts):,} samples, {total_en_replay_duration:.2f} hours")

    # Sample English replay to target hours
    print(f"   Sampling to {args.replay_hours:.2f} hours...")
    en_replay_sampled, en_replay_hours = sample_to_target_hours(en_replay_cuts, args.replay_hours)
    print(f"   Sampled: {len(en_replay_sampled):,} samples, {en_replay_hours:.2f} hours")

    # Combine all
    print("\n6. Combining and shuffling all data...")
    combined = ja_cv_sampled + en_cv_sampled + ja_replay_sampled + en_replay_sampled
    random.shuffle(combined)

    total_hours = sum(get_duration(c) for c in combined) / 3600
    total_ja_hours = ja_cv_hours + ja_replay_hours
    total_en_hours = en_cv_hours + en_replay_hours

    print(f"\nFinal manifest:")
    print(f"  Total: {len(combined):,} samples, {total_hours:.2f} hours")
    print(f"  Japanese: {len(ja_cv_sampled) + len(ja_replay_sampled):,} samples, {total_ja_hours:.2f} hours ({total_ja_hours/total_hours*100:.1f}%)")
    print(f"    - CommonVoice: {len(ja_cv_sampled):,} samples, {ja_cv_hours:.2f}h")
    print(f"    - Replay (ReazonSpeech): {len(ja_replay_sampled):,} samples, {ja_replay_hours:.2f}h")
    print(f"  English: {len(en_cv_sampled) + len(en_replay_sampled):,} samples, {total_en_hours:.2f} hours ({total_en_hours/total_hours*100:.1f}%)")
    print(f"    - CommonVoice: {len(en_cv_sampled):,} samples, {en_cv_hours:.2f}h")
    print(f"    - Replay (MLS): {len(en_replay_sampled):,} samples, {en_replay_hours:.2f}h")

    # Save
    output_file = output_dir / 'cv-bilingual_cuts_train.jsonl.gz'
    print(f"\n7. Saving to {output_file}...")
    with gzip.open(output_file, 'wt') as f:
        for cut in combined:
            f.write(json.dumps(cut, ensure_ascii=False) + '\n')

    print("Done!")


if __name__ == '__main__':
    main()
