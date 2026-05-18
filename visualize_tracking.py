"""
visualize_tracking.py
---------------------
Generira slike za poročilo:
  - Frame kjer tracker izgubi tarčo (is_lost=True) + samplovane regije
  - Frame kjer tracker znova najde tarčo (re-detection) + samplovane regije

Shrani slike v figures/ direktorij.

Uporaba:
    python visualize_tracking.py --dataset dataset/dataset-lt --net siamfc_net.pth
    python visualize_tracking.py --dataset dataset/dataset-lt --net siamfc_net.pth --sequence car9 --sampling_mode gaussian
"""

import argparse
import os
import cv2
import numpy as np
import torch

from tools.sequence_utils import VOTSequence
from siamfc_lt import TrackerSiamFCLT


def draw_box(img, box, color, thickness=2):
    """box = [x, y, w, h]"""
    x, y, w, h = [int(round(v)) for v in box]
    cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)


def draw_candidates(img, centers, patch_sz, color=(0, 165, 255), alpha=0.35):
    """
    Nariše kvadrate samplovanih regij na sliko.
    centers: (N, 2) array [y, x]
    patch_sz: velikost patch-a v pikslih
    """
    overlay = img.copy()
    half = patch_sz // 2
    for cy, cx in centers:
        cx, cy = int(round(cx)), int(round(cy))
        cv2.rectangle(overlay,
                      (cx - half, cy - half),
                      (cx + half, cy + half),
                      color, -1)
        cv2.rectangle(img,
                      (cx - half, cy - half),
                      (cx + half, cy + half),
                      color, 1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def run_and_capture(dataset_path, network_path, sequence_name,
                    sampling_mode='uniform', gaussian_std=None,
                    gaussian_std_growth=0.0,
                    threshold=4.0, n_samples=15,
                    out_dir='figures'):
    os.makedirs(out_dir, exist_ok=True)

    sequence = VOTSequence(dataset_path, sequence_name)
    tracker = TrackerSiamFCLT(
        net_path=network_path,
        failure_threshold=threshold,
        n_samples=n_samples,
        sampling_mode=sampling_mode,
        gaussian_std=gaussian_std,
        gaussian_std_growth=gaussian_std_growth,
    )

    img0 = cv2.imread(sequence.frame(0))
    gt_rect = sequence.get_annotation(0)
    tracker.init(img0, gt_rect)

    lost_frame_img = None
    redet_frame_img = None
    lost_idx = None
    redet_idx = None

    for i in range(1, sequence.length()):
        img = cv2.imread(sequence.frame(i))
        vis = img.copy()

        was_lost = tracker.is_lost

        # Pred update-om vzorčimo kandidate (samo za vizualizacijo)
        if was_lost:
            centers = tracker._sample_candidates(img)

        box, score = tracker.update(img)
        now_lost = tracker.is_lost

        if was_lost:
            # V re-detection načinu — nariši kandidate in predvideno pozicijo
            patch_sz = int(tracker.x_sz)
            draw_candidates(vis, centers, patch_sz,
                            color=(0, 165, 255))       # oranžna = samplovane regije
            draw_box(vis, box, color=(0, 0, 255), thickness=2)  # rdeča = napoved

            # Označi frame
            cv2.putText(vis, f'RE-DETECTION (score={score:.2f})',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

            if lost_frame_img is None:
                lost_frame_img = vis.copy()
                lost_idx = i
                print(f'  Saved LOST frame: {i}')

            if not now_lost and redet_frame_img is None:
                redet_frame_img = vis.copy()
                redet_idx = i
                print(f'  Saved RE-DETECTED frame: {i}')

        else:
            # Normalni tracking — nariši bbox zeleno
            draw_box(vis, box, color=(0, 200, 0), thickness=2)
            cv2.putText(vis, f'TRACKING (score={score:.2f})',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)

        # Zgodaj zaključi ko imamo oba primera
        if lost_frame_img is not None and redet_frame_img is not None:
            break

    # Shrani slike
    prefix = f'{out_dir}/{sequence_name}_{sampling_mode}'

    if lost_frame_img is not None:
        path = f'{prefix}_lost_frame{lost_idx}.png'
        cv2.imwrite(path, lost_frame_img)
        print(f'  Shranjeno: {path}')
    else:
        print('  ⚠ Ni bilo re-detection eventi v tej sekvenci.')

    if redet_frame_img is not None:
        path = f'{prefix}_redetected_frame{redet_idx}.png'
        cv2.imwrite(path, redet_frame_img)
        print(f'  Shranjeno: {path}')

    # Side-by-side slika za poročilo
    if lost_frame_img is not None and redet_frame_img is not None:
        h1, w1 = lost_frame_img.shape[:2]
        h2, w2 = redet_frame_img.shape[:2]
        h = max(h1, h2)

        # Resize če se razlikujeta
        if h1 != h:
            lost_frame_img = cv2.resize(lost_frame_img, (w1, h))
        if h2 != h:
            redet_frame_img = cv2.resize(redet_frame_img, (w2, h))

        side_by_side = np.hstack([lost_frame_img, redet_frame_img])

        # Dodaj naslove
        cv2.putText(side_by_side, f'Lost (frame {lost_idx})',
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(side_by_side, f'Re-detected (frame {redet_idx})',
                    (w1 + 10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        sbs_path = f'{prefix}_side_by_side.png'
        cv2.imwrite(sbs_path, side_by_side)
        print(f'  Shranjeno (side-by-side): {sbs_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',       required=True)
    parser.add_argument('--net',           required=True)
    parser.add_argument('--sequence',      default='car9',
                        help='Ime sekvence (privzeto: car9)')
    parser.add_argument('--sampling_mode', default='uniform',
                        choices=['uniform', 'gaussian', 'gaussian_growing'])
    parser.add_argument('--gaussian_std',  type=float, default=None)
    parser.add_argument('--gaussian_std_growth', type=float, default=10.0)
    parser.add_argument('--threshold',     type=float, default=4.0)
    parser.add_argument('--n_samples',     type=int,   default=15)
    parser.add_argument('--out_dir',       default='figures')
    args = parser.parse_args()

    print('=' * 60)
    print(f'  Vizualizacija: {args.sequence} | {args.sampling_mode}')
    print('=' * 60)

    run_and_capture(
        dataset_path=args.dataset,
        network_path=args.net,
        sequence_name=args.sequence,
        sampling_mode=args.sampling_mode,
        gaussian_std=args.gaussian_std,
        gaussian_std_growth=args.gaussian_std_growth,
        threshold=args.threshold,
        n_samples=args.n_samples,
        out_dir=args.out_dir,
    )

    print()
    print('Generiraj slike za oba načina samplinga:')
    print(f'  python visualize_tracking.py --dataset {args.dataset} --net {args.net} --sequence {args.sequence} --sampling_mode uniform')
    print(f'  python visualize_tracking.py --dataset {args.dataset} --net {args.net} --sequence {args.sequence} --sampling_mode gaussian')


if __name__ == '__main__':
    main()