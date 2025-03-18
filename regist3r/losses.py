from dust3r.losses import MultiLoss, Criterion, ConfLoss, L21, Sum, Regr3D_ScaleShiftInv


class Regr3DRegist3R(Criterion, MultiLoss):
    def __init__(self, criterion, norm_mode='avg_dis', gt_scale=False):
        super().__init__(criterion)
        self.norm_mode = norm_mode
        self.gt_scale = gt_scale
    
    def compute_loss(self, gt1, gt2, pred1, pred2, **kw):
        gt_pts1 = pred1['gt_pts3d']
        gt_pts2 = pred2['gt_pts3d']
        pr_pts1 = pred1['pts3d']
        pr_pts2 = pred2['pts3d']
        valid1 = gt1['valid_mask']
        valid2 = gt2['valid_mask']

        l1 = self.criterion(pr_pts1[valid1], gt_pts1[valid1])
        l2 = self.criterion(pr_pts2[valid2], gt_pts2[valid2])
        self_name = type(self).__name__
        details = {self_name + '_pts3d_1': l1.mean().item(), self_name + '_pts3d_2': l2.mean().item()}

        return Sum((l1, valid1), (l2, valid2)), details
