"""
test_n_samples.py
-----------------
Zažene SiamFC-LT tracker za več vrednosti n_samples
s fiksnim failure_threshold=3.3 in za vsako izpiše
Precision, Recall ter F-score.

Uporaba:
    python test_n_samples.py --dataset dataset/dataset-lt --net siamfc_net.pth

Opcijsko:
    --results_base  kje shrani rezultate  (privzeto: results_n_samples)
    --threshold     fiksni threshold      (privzeto: 3.3)
"""

import argparse
import os
import re
import subprocess
import sys

import cv2

from tools.sequence_utils import VOTSequence, save_results
from siamfc_lt import TrackerSiamFCLT


# ── helpers ──────────────────────────────────────────────────────────────────

def run_tracker(dataset_path, network_path, results_dir, threshold, n_samples, visualize=False):
    """Poženi tracker z določenim n_samples in threshold-om na vseh sekvencah."""
    os.makedirs(results_dir, exist_ok=True)

    sequences = []
    with open(os.path.join(dataset_path, 'list.txt'), 'r') as f:
        for line in f.readlines():
            sequences.append(line.strip())

    tracker = TrackerSiamFCLT(
        net_path=network_path,
        failure_threshold=threshold,
        n_samples=n_samples,
    )

    for sequence_name in sequences:
        print(f'  [tracker] Sekvenca: {sequence_name}')

        bboxes_path = os.path.join(results_dir, f'{sequence_name}_bboxes.txt')
        scores_path = os.path.join(results_dir, f'{sequence_name}_scores.txt')

        if os.path.exists(bboxes_path) and os.path.exists(scores_path):
            print('    Rezultati že obstajajo. Preskakujem.')
            continue

        sequence = VOTSequence(dataset_path, sequence_name)

        img = cv2.imread(sequence.frame(0))
        gt_rect = sequence.get_annotation(0)
        tracker.init(img, gt_rect)

        results = [gt_rect]
        scores = [[10000]]

        if visualize:
            cv2.namedWindow('win', cv2.WINDOW_AUTOSIZE)

        for i in range(1, sequence.length()):
            img = cv2.imread(sequence.frame(i))
            prediction, score = tracker.update(img)
            results.append(prediction)
            scores.append([score])

            if visualize:
                tl_ = (int(round(prediction[0])), int(round(prediction[1])))
                br_ = (int(round(prediction[0] + prediction[2])),
                       int(round(prediction[1] + prediction[3])))
                cv2.rectangle(img, tl_, br_, (0, 0, 255), 1)
                cv2.imshow('win', img)
                if cv2.waitKey(10) == 27:
                    sys.exit(0)

        save_results(results, bboxes_path)
        save_results(scores, scores_path)


def run_evaluation(dataset_path, results_dir):
    """Pokliče performance_evaluation.py in vrne (precision, recall, fscore)."""
    cmd = [
        sys.executable, 'performance_evaluation.py',
        '--dataset', dataset_path,
        '--results_dir', results_dir,
    ]
    out = subprocess.run(cmd, capture_output=True, text=True)
    output = out.stdout + out.stderr

    precision = recall = fscore = None
    m = re.search(r'Precision:\s*([\d.]+)', output)
    if m:
        precision = float(m.group(1))
    m = re.search(r'Recall:\s*([\d.]+)', output)
    if m:
        recall = float(m.group(1))
    m = re.search(r'F-score:\s*([\d.]+)', output)
    if m:
        fscore = float(m.group(1))

    return precision, recall, fscore, output.strip()


def fmt(val):
    return f'{val:.4f}' if val is not None else 'N/A'


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Eksperimenti z različnimi n_samples vrednostmi')
    parser.add_argument('--dataset',      required=True,  help='Pot do dataseta')
    parser.add_argument('--net',          required=True,  help='Pot do siamfc_net.pth')
    parser.add_argument('--results_base', default='results_n_samples',
                                          help='Korenski direktorij za rezultate')
    parser.add_argument('--threshold',    type=float, default=4.0,
                                          help='Fiksni failure_threshold (privzeto: 3.3)')
    parser.add_argument('--visualize',    action='store_true', help='Pokaži tracking okno')
    args = parser.parse_args()

    # Vrednosti N za testiranje
    n_values = [1, 5, 10, 15, 20, 25, 50, 100]

    print('=' * 65)
    print('  SiamFC-LT — eksperimenti z različnimi n_samples')
    print('=' * 65)
    print(f'  Fiksni threshold : {args.threshold}')
    print(f'  Testirani N      : {n_values}')
    print()

    results_summary = []

    for n in n_values:
        results_dir = os.path.join(args.results_base, f'n_samples_{n}')

        print('-' * 65)
        print(f'  Instanca: n_samples = {n}  (threshold = {args.threshold})')
        print(f'  Results dir: {results_dir}')
        print()

        # 1) Zaženi tracker
        print('  [1/2] Zaganjam tracker ...')
        try:
            run_tracker(args.dataset, args.net, results_dir, args.threshold, n, args.visualize)
        except Exception as e:
            print(f'  ⚠ Napaka pri trackerju: {e}')
            results_summary.append((n, None, None, None))
            print()
            continue

        # 2) Evaluacija
        print('  [2/2] Evalviram rezultate ...')
        precision, recall, fscore, raw = run_evaluation(args.dataset, results_dir)
        print(f'  {raw}')
        print()

        results_summary.append((n, precision, recall, fscore))

    # ── Povzetna tabela ───────────────────────────────────────────────────────
    print()
    print('=' * 65)
    print('  POVZETEK REZULTATOV')
    print('=' * 65)
    header = f"{'N (samples)':>12}  {'Precision':>10} {'Recall':>8} {'F-score':>8}"
    print(header)
    print('-' * 45)
    for n, p, r, f in results_summary:
        print(f'{n:>12}  {fmt(p):>10} {fmt(r):>8} {fmt(f):>8}')

    print()
    print(f'Rezultati shranjeni v: {args.results_base}/')
    print(f'Threshold uporabljen : {args.threshold}')


if __name__ == '__main__':
    main()