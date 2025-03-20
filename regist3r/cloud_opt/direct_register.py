import torch
import numpy as np
import cv2
import tqdm

from dust3r.viz import rgb
from dust3r.cloud_opt.base_opt import BasePCOptimizer
from dust3r.post_process import estimate_focal_knowing_depth
from dust3r.utils.geometry import inv, geotrf


class DirectRegister(BasePCOptimizer):
    def __init__(self, view, pred, indices, ref_indices):
        super(BasePCOptimizer, self).__init__()
        self.n_imgs = len(view['img'])
        self.edges = [(int(view['idx'][i]), int(view['idx'][i+1])) for i in range(self.n_imgs-1)]
        self.imshapes = [view['img'][i].shape[-2:] for i in range(self.n_imgs)]
        self.conf_trf = lambda x : x.log()
        self.im_conf = [v for v in pred['conf']]
        self.min_conf_thr = 3

        # possibly store images for show_pointcloud
        self.imgs = None
        if 'img' in view:
            imgs = [v for v in view['img']]
            self.imgs = rgb(imgs)
        
        self.pts3d = [p for p in pred['pts3d']]
        self.pp = [torch.tensor((W/2, H/2), device=self.device) for H, W in self.imshapes]

        start = -1
        for i, ir in zip(indices, ref_indices):
            if i == ir:
                start = i
                break
        assert start >= 0

        focal = float(estimate_focal_knowing_depth(self.pts3d[start][None], self.pp[start], focal_mode='weiszfeld'))

        self.focals = [focal for _ in self.imshapes]
        self.im_poses = []
        self.depth = []
        # use dust3r pts1 to compute focal, and share this focal among all frames.
        for i in tqdm.trange(0, self.n_imgs, desc="PnP Solving"):
            H, W = self.imshapes[i]
            pts3d = self.pts3d[i].cpu().numpy()
            pixels = np.mgrid[:W, :H].T.astype(np.float32)
            assert pts3d.shape[:2] == (H, W)
            msk = self.get_masks()[i].cpu().numpy()
            K = np.float32([(focal, 0, W/2), (0, focal, H/2), (0, 0, 1)])

            try:
                res = cv2.solvePnPRansac(pts3d[msk], pixels[msk], K, None,
                                         iterationsCount=100, reprojectionError=5, flags=cv2.SOLVEPNP_SQPNP)
                success, R, T, inliers = res
                assert success

                R = cv2.Rodrigues(R)[0]
                pose = inv(np.r_[np.c_[R, T], [(0, 0, 0, 1)]])
            except:
                pose = np.eye(4)
            rel_pose = torch.from_numpy(pose.astype(np.float32)).to(self.device)
            self.im_poses.append(rel_pose)
            self.depth.append(geotrf(inv(rel_pose), self.pts3d[i])[..., 2])

        self.im_poses = torch.stack(self.im_poses, dim=0)
        self.im_conf = torch.stack(self.im_conf, dim=0)
        self.im_conf_display = self.im_conf.clone()
        self.focals = torch.tensor(self.focals, device=self.device)
        self.depth = torch.stack(self.depth, dim=0)

        self.indices = indices
        self.ref_indices = ref_indices

    def _init_from_views(self, view, pred):
        super().__init__()

    def get_focals(self):
        return self.focals

    def get_im_poses(self):
        return self.im_poses

    def get_pts3d(self):
        return self.pts3d

    def get_depthmaps(self, raw=False):
        return [d.to(self.device) for d in self.depth]

    def get_principal_points(self):
        return self.pp
    
    def get_intrinsics(self):
        focals = self.get_focals()
        pps = self.get_principal_points()
        K = torch.zeros((len(focals), 3, 3), device=self.device)
        for i in range(len(focals)):
            K[i, 0, 0] = K[i, 1, 1] = focals[i]
            K[i, :2, 2] = pps[i]
            K[i, 2, 2] = 1
        return K
    
    def clean_pointcloud(self, **kw):
        print("enter clean_pointcloud")
        super().clean_pointcloud(**kw)
        print("leave clean_pointcloud")
        return self

    @property
    def device(self):
        return self.pts3d[0].device
