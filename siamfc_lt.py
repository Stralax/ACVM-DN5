import numpy as np
import torch
import cv2

from siamfc import TrackerSiamFC
from siamfc import ops  # ali: from ops import crop_and_resize


class TrackerSiamFCLT(TrackerSiamFC):

    def __init__(self, net_path=None, 
                 failure_threshold=3.3,
                 n_samples=25,
                 **kwargs):
        super().__init__(net_path=net_path, **kwargs)
        self.failure_threshold = failure_threshold
        self.n_samples = n_samples
        self.is_lost = False

    @torch.no_grad()
    def update(self, img):
        # --- normalni short-term update ---
        box, max_resp = super().update(img)
        # print(f'max_resp={max_resp:.4f}, is_lost={self.is_lost}')

        if not self.is_lost:
            if max_resp < self.failure_threshold:
                # tarča izgubljena
                self.is_lost = True
                # vrni zadnjo znano pozicijo z nizkim scoreom
                return box, max_resp
            else:
                return box, max_resp

        else:
            # --- RE-DETECTION MODE ---
            h, w = img.shape[:2]

            # Naključno vzorči n_samples pozicij po celi sliki
            ys = np.random.uniform(0, h, self.n_samples)
            xs = np.random.uniform(0, w, self.n_samples)
            centers = np.stack([ys, xs], axis=1)  # (N, 2) v [y, x]

            best_score = -np.inf
            best_box = box  # fallback

            # Za vsak vzorec: izreži patch in izračunaj response
            patches = []
            for cy, cx in centers:
                center = np.array([cy, cx])
                patch = ops.crop_and_resize(
                    img, center, self.x_sz,
                    out_size=self.cfg.instance_sz,
                    border_value=self.avg_color)
                patches.append(patch)

            patches = np.stack(patches, axis=0)  # (N, H, W, 3)
            patches_t = torch.from_numpy(patches).to(
                self.device).permute(0, 3, 1, 2).float()

            # Backbone features za vse patche naenkrat
            feats = self.net.backbone(patches_t)
            # Response za vsak patch
            responses = self.net.head(self.kernel, feats)
            responses = responses.squeeze(1).cpu().numpy()

            for i, (cy, cx) in enumerate(centers):
                resp = responses[i]
                score = float(resp.max())
                if score > best_score:
                    best_score = score
                    # Posodobi center
                    # (za preprostost: patch center = nova pozicija)
                    best_center = np.array([cy, cx])
                    best_box = np.array([
                        cx + 1 - (self.target_sz[1] - 1) / 2,
                        cy + 1 - (self.target_sz[0] - 1) / 2,
                        self.target_sz[1], self.target_sz[0]])

            if best_score > self.failure_threshold:
                # Tarča najdena — obnovi center
                self.center = best_center
                self.is_lost = False

            return best_box, best_score