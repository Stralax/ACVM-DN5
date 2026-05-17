import numpy as np
import torch
import cv2

from siamfc import TrackerSiamFC
from siamfc import ops


class TrackerSiamFCLT(TrackerSiamFC):

    def __init__(self, net_path=None,
                 failure_threshold=3.3,
                 n_samples=25,
                 use_dynamic_threshold=False,
                 dynamic_threshold_ratio=0.5,
                 **kwargs):
        """
        use_dynamic_threshold:    če True, ignorira failure_threshold in izračuna
                                  prag iz prvega frame-a kot:
                                  threshold = dynamic_threshold_ratio * initial_score
        dynamic_threshold_ratio:  delež začetnega scora (npr. 0.5 = 1/2, 0.333 = 1/3)
        """
        super().__init__(net_path=net_path, **kwargs)
        self.failure_threshold = failure_threshold
        self.n_samples = n_samples
        self.is_lost = False

        self.use_dynamic_threshold = use_dynamic_threshold
        self.dynamic_threshold_ratio = dynamic_threshold_ratio
        self._initial_score = None  # bo nastavljeno ob prvem update()

    @torch.no_grad()
    def update(self, img):
        # --- normalni short-term update ---
        box, max_resp = super().update(img)

        # Nastavi dinamični threshold ob prvem klicu update()
        if self.use_dynamic_threshold and self._initial_score is None:
            self._initial_score = max_resp
            self.failure_threshold = self.dynamic_threshold_ratio * self._initial_score
            print(f'  [dynamic threshold] initial_score={self._initial_score:.4f}, '
                  f'ratio={self.dynamic_threshold_ratio}, '
                  f'threshold={self.failure_threshold:.4f}')

        if not self.is_lost:
            if max_resp < self.failure_threshold:
                self.is_lost = True
                return box, max_resp
            else:
                return box, max_resp

        else:
            # --- RE-DETECTION MODE ---
            h, w = img.shape[:2]

            ys = np.random.uniform(0, h, self.n_samples)
            xs = np.random.uniform(0, w, self.n_samples)
            centers = np.stack([ys, xs], axis=1)

            best_score = -np.inf
            best_box = box

            patches = []
            for cy, cx in centers:
                center = np.array([cy, cx])
                patch = ops.crop_and_resize(
                    img, center, self.x_sz,
                    out_size=self.cfg.instance_sz,
                    border_value=self.avg_color)
                patches.append(patch)

            patches = np.stack(patches, axis=0)
            patches_t = torch.from_numpy(patches).to(
                self.device).permute(0, 3, 1, 2).float()

            feats = self.net.backbone(patches_t)
            responses = self.net.head(self.kernel, feats)
            responses = responses.squeeze(1).cpu().numpy()

            for i, (cy, cx) in enumerate(centers):
                resp = responses[i]
                score = float(resp.max())
                if score > best_score:
                    best_score = score
                    best_center = np.array([cy, cx])
                    best_box = np.array([
                        cx + 1 - (self.target_sz[1] - 1) / 2,
                        cy + 1 - (self.target_sz[0] - 1) / 2,
                        self.target_sz[1], self.target_sz[0]])

            if best_score > self.failure_threshold:
                self.center = best_center
                self.is_lost = False

            return best_box, best_score