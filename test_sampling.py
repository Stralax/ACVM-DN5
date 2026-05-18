"""
test_sampling.py
----------------
Primerja uniform, gaussian (fiksni std) in gaussian_growing sampling
pri re-detection. Threshold=4.0, N=15.

Uporaba:
    python test_sampling.py --dataset dataset/dataset-lt --net siamfc_net.pth
"""

import argparse
import os
import re
import subprocess
import sys
import cv2

from tools.sequence_utils import VOTSequence, save_results
from siamfc_lt import TrackerSiamFCLT


def run_tracker(dataset_path, network_path, results_dir,
                threshold=4.0, n_samples=15,
                sampling_mode='uniform',
                gaussian_std=None,
                gaussian_std_growth=0.0,
                visualize=False):
    os.makedirs(results_dir, exist_ok=True)

    sequences = []
    with open(os.path.join(dataset_path, 'list.txt'), 'r') as f:
        for line in f.readlines():
            sequences.append(line.strip())

    for sequence_name in sequences:
        print(f'  [tracker] Sekvenca: {sequence_name}')

        bboxes_path = os.path.join(results_dir, f'{sequence_name}_bboxes.txt')
        scores_path = os.path.join(results_dir, f'{sequence_name}_scores.txt')

        if os.path.exists(bboxes_path) and os.path.exists(scores_path):
            print('    Rezultati že obstajajo. Preskakujem.')
            continue

        tracker = TrackerSiamFCLT(
            net_path=network_path,
            failure_threshold=threshold,
            n_samples=n_samples,
            sampling_mode=sampling_mode,
            gaussian_std=gaussian_std,
            gaussian_std_growth=gaussian_std_growth,
        )

        sequence = VOTSequence(dataset_path, sequence_name)
        img = cv2.imread(sequence.frame(0))
        gt_rect = sequence.get_annotation(0)
        tracker.init(img, gt_rect)

        results = [gt_rect]
        scores = [[10000]]

        for i in range(1, sequence.length()):
            img = cv2.imread(sequence.frame(i))
            prediction, score = tracker.update(img)
            results.append(prediction)
            scores.append([score])

        save_results(results, bboxes_path)
        save_results(scores, scores_path)


def run_evaluation(dataset_path, results_dir):
    cmd = [sys.executable, 'performance_evaluation.py',
           '--dataset', dataset_path, '--results_dir', results_dir]
    out = subprocess.run(cmd, capture_output=True, text=True)
    output = out.stdout + out.stderr

    precision = recall = fscore = None
    m = re.search(r'Precision:\s*([\d.]+)', output)
    if m: precision = float(m.group(1))
    m = re.search(r'Recall:\s*([\d.]+)', output)
    if m: recall = float(m.group(1))
    m = re.search(r'F-score:\s*([\d.]+)', output)
    if m: fscore = float(m.group(1))

    return precision, recall, fscore, output.strip()


def fmt(val):
    return f'{val:.4f}' if val is not None else 'N/A'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',      required=True)
    parser.add_argument('--net',          required=True)
    parser.add_argument('--results_base', default='results_sampling')
    parser.add_argument('--threshold',    type=float, default=4.0)
    parser.add_argument('--n_samples',    type=int,   default=15)
    args = parser.parse_args()

    # (label, dir_suffix, kwargs za run_tracker)
    experiments = [
        ('Uniform',
         'uniform',
         dict(sampling_mode='uniform')),

        ('Gaussian fixed std=50',
         'gaussian_std50',
         dict(sampling_mode='gaussian', gaussian_std=50)),

        ('Gaussian fixed std=100',
         'gaussian_std100',
         dict(sampling_mode='gaussian', gaussian_std=100)),

        ('Gaussian fixed std=200',
         'gaussian_std200',
         dict(sampling_mode='gaussian', gaussian_std=200)),

        ('Gaussian auto std (min(h,w)/6)',
         'gaussian_auto',
         dict(sampling_mode='gaussian', gaussian_std=None)),

        ('Gaussian growing std=50 +10/frame',
         'gaussian_grow50_10',
         dict(sampling_mode='gaussian_growing', gaussian_std=50, gaussian_std_growth=10)),

        ('Gaussian growing std=50 +20/frame',
         'gaussian_grow50_20',
         dict(sampling_mode='gaussian_growing', gaussian_std=50, gaussian_std_growth=20)),
    ]

    print('=' * 70)
    print('  SiamFC-LT — primerjava sampling strategij')
    print('=' * 70)
    print(f'  Threshold : {args.threshold}   N : {args.n_samples}')
    print()

    results_summary = []

    for label, suffix, kwargs in experiments:
        results_dir = os.path.join(args.results_base, suffix)

        print('-' * 70)
        print(f'  Instanca : {label}')
        print(f'  Results  : {results_dir}')
        print()

        print('  [1/2] Zaganjam tracker ...')
        try:
            run_tracker(args.dataset, args.net, results_dir,
                        threshold=args.threshold,
                        n_samples=args.n_samples,
                        **kwargs)
        except Exception as e:
            print(f'  ⚠ Napaka: {e}')
            results_summary.append((label, None, None, None))
            print()
            continue

        print('  [2/2] Evalviram ...')
        precision, recall, fscore, raw = run_evaluation(args.dataset, results_dir)
        print(f'  {raw}')
        print()
        results_summary.append((label, precision, recall, fscore))

    print()
    print('=' * 70)
    print('  POVZETEK REZULTATOV')
    print('=' * 70)
    print(f"{'Strategija':<40} {'Precision':>10} {'Recall':>8} {'F-score':>8}")
    print('-' * 70)
    for label, p, r, f in results_summary:
        print(f'{label:<40} {fmt(p):>10} {fmt(r):>8} {fmt(f):>8}')

    print()
    print(f'Rezultati shranjeni v: {args.results_base}/')


if __name__ == '__main__':
    main()