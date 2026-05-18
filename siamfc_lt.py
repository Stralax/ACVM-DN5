import numpy as np
import torch
import cv2

from siamfc import TrackerSiamFC
from siamfc import ops


class TrackerSiamFCLT(TrackerSiamFC):

    def __init__(self, net_path=None,
                 failure_threshold=4.0,
                 n_samples=15,
                 use_dynamic_threshold=False,
                 dynamic_threshold_ratio=0.5,
                 sampling_mode='uniform',
                 gaussian_std=None,
                 gaussian_std_growth=0.0,
                 **kwargs):
        """
        sampling_mode:
            'uniform'  — enakomerni naključni sampling po celi sliki
            'gaussian' — Gaussov sampling okoli zadnje znane pozicije
            'gaussian_growing' — Gaussov sampling z naraščajočim std

        gaussian_std:
            začetna standardna deviacija v pikslih.
            Če None → auto: min(h, w) / 6  (nastavi ob prvem re-detection klicu)

        gaussian_std_growth:
            pikslov ki se prištejejo k std po vsakem re-detection frame-u.
            Ignorira se pri 'uniform' in 'gaussian'.
        """
        super().__init__(net_path=net_path, **kwargs)
        self.failure_threshold = failure_threshold
        self.n_samples = n_samples
        self.is_lost = False

        self.use_dynamic_threshold = use_dynamic_threshold
        self.dynamic_threshold_ratio = dynamic_threshold_ratio
        self._initial_score = None

        self.sampling_mode = sampling_mode
        self._gaussian_std_init = gaussian_std   # None = auto
        self.gaussian_std = gaussian_std
        self.gaussian_std_growth = gaussian_std_growth
        self._redetect_frames = 0

    # ------------------------------------------------------------------
    def _sample_candidates(self, img):
        """Vrne (N, 2) array centrov [y, x] glede na sampling_mode."""
        h, w = img.shape[:2]

        if self.sampling_mode == 'uniform':
            ys = np.random.uniform(0, h, self.n_samples)
            xs = np.random.uniform(0, w, self.n_samples)

        elif self.sampling_mode in ('gaussian', 'gaussian_growing'):
            # Auto std ob prvem klicu
            if self.gaussian_std is None:
                self.gaussian_std = min(h, w) / 6.0

            if self.sampling_mode == 'gaussian_growing':
                current_std = self.gaussian_std + self.gaussian_std_growth * self._redetect_frames
            else:
                current_std = self.gaussian_std

            cy, cx = self.center   # [y, x] zadnja znana pozicija
            ys = np.random.normal(cy, current_std, self.n_samples)
            xs = np.random.normal(cx, current_std, self.n_samples)

            # Clipaj na meje slike
            ys = np.clip(ys, 0, h - 1)
            xs = np.clip(xs, 0, w - 1)

        else:
            raise ValueError(f"Neznan sampling_mode: '{self.sampling_mode}'. "
                             "Uporabi 'uniform', 'gaussian' ali 'gaussian_growing'.")

        return np.stack([ys, xs], axis=1)  # (N, 2)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def update(self, img):
        box, max_resp = super().update(img)

        # Dinamični threshold ob prvem klicu
        if self.use_dynamic_threshold and self._initial_score is None:
            self._initial_score = max_resp
            self.failure_threshold = self.dynamic_threshold_ratio * self._initial_score
            print(f'  [dynamic threshold] initial_score={self._initial_score:.4f}, '
                  f'ratio={self.dynamic_threshold_ratio}, '
                  f'threshold={self.failure_threshold:.4f}')

        if not self.is_lost:
            if max_resp < self.failure_threshold:
                self.is_lost = True
                self._redetect_frames = 0
                self.gaussian_std = self._gaussian_std_init  # reset std
                return box, max_resp
            else:
                return box, max_resp

        else:
            # RE-DETECTION MODE
            centers = self._sample_candidates(img)
            self._redetect_frames += 1

            best_score = -np.inf
            best_box = box
            best_center = None

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
                score = float(responses[i].max())
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
                self._redetect_frames = 0
                self.gaussian_std = self._gaussian_std_init  # reset std

            return best_box, best_score