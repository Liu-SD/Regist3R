import torch
from torch import nn
import torch.distributed
import torch.distributed
from torch.utils import checkpoint
from croco.models.croco import CroCoNet
from croco.models.blocks import Block, DecoderBlock
from copy import deepcopy
from dust3r.utils.misc import fill_default_args, freeze_all_params, transpose_to_landscape
from dust3r.utils.geometry import geotrf
from dust3r.heads import head_factory
from dust3r.patch_embed import PatchEmbedDust3R, ManyAR_PatchEmbed

inf = float('inf')

def get_patch_embed(patch_embed_cls, img_size, patch_size, enc_embed_dim, channel=3):
    assert patch_embed_cls in ['PatchEmbedDust3R', 'ManyAR_PatchEmbed']
    patch_embed = eval(patch_embed_cls)(img_size, patch_size, channel, enc_embed_dim)
    return patch_embed

def load_model(model_path, device, verbose=True):
    if verbose:
        print('... loading model from', model_path)
    ckpt = torch.load(model_path, map_location='cpu')
    args = ckpt['args'].model.replace("ManyAR_PatchEmbed", "PatchEmbedDust3R")
    if 'landscape_only' not in args:
        args = args[:-1] + ', landscape_only=False)'
    else:
        args = args.replace(" ", "").replace('landscape_only=True', 'landscape_only=False')
    assert "landscape_only=False" in args
    if verbose:
        print(f"instantiating : {args}")
    net = eval(args)
    s = net.load_state_dict(ckpt['model'], strict=False)
    if verbose:
        print(s)
    return net.to(device)


class Regist3R(CroCoNet):
    def __init__(self,
                 output_mode='pts3d',
                 head_type='linear',
                 depth_mode=('exp', -inf, inf),
                 conf_mode=('exp', 1, inf),
                 freeze='none',
                 landscape_only=True,
                 patch_embed_cls='PatchEmbedDust3R',  # PatchEmbedDust3R or ManyAR_PatchEmbed
                 use_checkpoint=False,
                 **croco_kwargs):
        self.patch_embed_cls = patch_embed_cls
        self.croco_args = fill_default_args(croco_kwargs, super().__init__)
        super().__init__(**croco_kwargs)

        self.enc_blocks_ref = deepcopy(self.enc_blocks)
        self.enc_norm_ref = deepcopy(self.enc_norm)
        self.set_downstream_head(output_mode, head_type, landscape_only, depth_mode, conf_mode, **croco_kwargs)
        self.set_freeze(freeze)

        self.use_checkpoint = use_checkpoint
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kw):
        return load_model(pretrained_model_name_or_path, device='cpu')

    def _set_decoder(self, enc_embed_dim, dec_embed_dim, dec_num_heads, dec_depth, mlp_ratio, norm_layer, norm_im2_in_dec):
        self.dec_depth = dec_depth
        self.dec_embed_dim = dec_embed_dim
        # transfer from encoder to decoder 
        self.decoder_embed = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)
        # transformer for the decoder 
        self.dec_blocks = nn.ModuleList([
            DecoderBlock(dec_embed_dim, dec_num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer, norm_mem=norm_im2_in_dec, rope=self.rope)
            for i in range(dec_depth)])
        # transfer from encoder to decoder 
        self.decoder_embed_ref = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)
        # transformer for the decoder_ref
        self.dec_blocks_ref = nn.ModuleList([
            Block(dec_embed_dim, dec_num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer, rope=self.rope)
            for i in range(dec_depth-1)])
        # final norm layer 
        self.dec_norm = norm_layer(dec_embed_dim)

    def set_downstream_head(self, output_mode, head_type, landscape_only, depth_mode, conf_mode, patch_size, img_size,
                            **kw):
        assert img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0, \
            f'{img_size=} must be multiple of {patch_size=}'
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        # allocate heads
        self.downstream_head = head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        # magic wrapper
        self.head = transpose_to_landscape(self.downstream_head, activate=landscape_only)

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768):
        self.patch_embed = get_patch_embed(self.patch_embed_cls, img_size, patch_size, enc_embed_dim, channel=3)
        # 0-2: RGB, 3-5: pointmap, 6: confidence
        self.patch_embed_ref = get_patch_embed(self.patch_embed_cls, img_size, patch_size, enc_embed_dim, channel=3+3+1)

    def set_freeze(self, freeze):  # this is for use by downstream models
        self.freeze = freeze
        to_be_frozen = {
            'none': [],
            'mask': [self.mask_token],
            'encoder': [self.mask_token, self.patch_embed, self.enc_blocks],
        }
        freeze_all_params(to_be_frozen[freeze])
    
    def load_state_dict(self, ckpt, **kw):
        # duplicate all weights for the second encoder if not present
        ckpt = {k: v for k, v in ckpt.items() if not k.startswith('dec_blocks2')}
        # all heads are trained from scratch
        ckpt = {k: v for k, v in ckpt.items() if not k.startswith('downstream_head1')}
        ckpt = {k: v for k, v in ckpt.items() if not k.startswith('downstream_head2')}
        new_ckpt = dict(ckpt)
        if not any(k.startswith('enc_blocks_ref') for k in ckpt):
            for key, value in ckpt.items():
                if key.startswith('enc_blocks'):
                    new_ckpt[key.replace('enc_blocks', 'enc_blocks_ref')] = value
        if not any(k.startswith('dec_blocks_ref') for k in ckpt):
            for key, value in ckpt.items():
                if key.startswith('dec_blocks'):
                    new_ckpt[key.replace('dec_blocks', 'dec_blocks_ref')] = value
        return super().load_state_dict(new_ckpt, **kw)
    
    def preprocess_pts(self, view1, view2):
        gt_pts1 = view1['pts3d'].permute(0, 3, 1, 2) # B,3,H,W
        pr_pts1 = view1['pr_pts3d'].permute(0, 3, 1, 2)
        gt_pts2 = view2['pts3d'].permute(0, 3, 1, 2)
        valid1 = view1['valid_mask'].unsqueeze(1) # B,1,H,W
        valid2 = view2['valid_mask'].unsqueeze(1)
        gt_pts1 = torch.masked_fill(gt_pts1, ~valid1, torch.nan)
        pr_pts1 = torch.masked_fill(pr_pts1, ~valid1, torch.nan)
        gt_pts2 = torch.masked_fill(gt_pts2, ~valid2, torch.nan)

        mean_v1 = gt_pts1.nanmean(dim=(2,3), keepdim=True)
        mean_v2 = gt_pts2.nanmean(dim=(2,3), keepdim=True)
        avg_dis_v1 = torch.norm(gt_pts1 - mean_v1, dim=1, keepdim=True).nanmean(dim=(2,3), keepdim=True)
        avg_dis_v2 = torch.norm(gt_pts2 - mean_v2, dim=1, keepdim=True).nanmean(dim=(2,3), keepdim=True)

        pr_pts1_normed_by_view1 = (pr_pts1 - mean_v1) / (avg_dis_v1 + 1e-8)
        gt_pts2_normed_by_view1 = (gt_pts2 - mean_v1) / (avg_dis_v1 + 1e-8)
        gt_pts1_normed_by_view2 = (gt_pts1 - mean_v2) / (avg_dis_v2 + 1e-8)
        gt_pts2_normed_by_view2 = (gt_pts2 - mean_v2) / (avg_dis_v2 + 1e-8)

        pr_pts1_normed_by_view1.nan_to_num_(0.)
        gt_pts2_normed_by_view1.nan_to_num_(0.)
        gt_pts1_normed_by_view2.nan_to_num_(0.)
        gt_pts2_normed_by_view2.nan_to_num_(0.)

        return pr_pts1_normed_by_view1, gt_pts2_normed_by_view1, gt_pts1_normed_by_view2, gt_pts2_normed_by_view2, \
                (mean_v1, avg_dis_v1, mean_v2, avg_dis_v2)
    
    def _encode_image(self, image, patch_embed, enc_blocks, enc_norm, true_shape):
        x, pos = patch_embed(image, true_shape=true_shape)
        for blk in enc_blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, pos)
            else:
                x = blk(x, pos)
        x = enc_norm(x)
        return x, pos, None
    
    def _encode_image_pairs(self, view_ref, view, pts_ref, cached_feat=None):
        img_ref = view_ref['img']
        img = view['img']
        if 'conf' in view_ref:
            conf = view_ref['conf']
        else:
            conf = torch.ones_like(view_ref['img'][:,0])
        img_ref = torch.cat((img_ref, pts_ref, conf.unsqueeze(1)), dim=1)
        B = img.shape[0]
        # Recover true_shape when available, otherwise assume that the img shape is the true one
        shape_ref = view_ref.get('true_shape', torch.tensor(img_ref.shape[-2:])[None].repeat(B, 1))
        shape = view.get('true_shape', torch.tensor(img.shape[-2:])[None].repeat(B, 1))

        feat_ref, pos_ref, _ = self._encode_image(img_ref, self.patch_embed_ref, self.enc_blocks_ref, self.enc_norm_ref, shape_ref)
        if cached_feat is not None:
            feat, pos = cached_feat
        else:
            with torch.no_grad():
                feat, pos, _ = self._encode_image(img, self.patch_embed, self.enc_blocks, self.enc_norm, shape)

        return (shape_ref, shape), (feat_ref, feat), (pos_ref, pos)

    def _decoder(self, feat_ref, pos_ref, feat, pos):
        final_output = [feat]
        feat = self.decoder_embed(feat)
        feat_ref = self.decoder_embed_ref(feat_ref)

        ref_output = [feat_ref]
        for blk in self.dec_blocks_ref:
            if self.use_checkpoint:
                feat_ref = checkpoint.checkpoint(blk, ref_output[-1], pos_ref)
            else:
                feat_ref = blk(ref_output[-1], pos_ref)
            ref_output.append(feat_ref)

        final_output.append(feat)
        for i, blk in enumerate(self.dec_blocks):
            if self.use_checkpoint:
                feat, _ = checkpoint.checkpoint(blk, final_output[-1], ref_output[i], pos, pos_ref)
            else:
                feat, _ = blk(final_output[-1], ref_output[i], pos, pos_ref)
            final_output.append(feat)
        final_output[-1] = self.dec_norm(final_output[-1])
        final_output.pop(1)
        return final_output
    
    def _forward(self, view_ref, pts_ref, view, cached_feat=None):
        (shape_ref, shape), (feat_ref, feat), (pos_ref, pos) = self._encode_image_pairs(view_ref, view, pts_ref, cached_feat=cached_feat)
        dec = self._decoder(feat_ref, pos_ref, feat, pos)
        res = self.head(dec, shape)
        return res, (feat, pos)
    
    def forward(self, view1, view2):
        # get normalized pointmap. mean=0, std=1, random rotation for P1.
        pr_pts1_normed_by_view1, gt_pts2_normed_by_view1, gt_pts1_normed_by_view2, gt_pts2_normed_by_view2, stats \
            = self.preprocess_pts(view1=view1, view2=view2)

        # view1 as reference view, predict view2's pts3d w.r.t. gt_pts1_normed_by_view1
        res2, cached_feat2 = self._forward(view1, pr_pts1_normed_by_view1, view2)
        res2['gt_pts3d'] = gt_pts2_normed_by_view1.permute(0,2,3,1)

        # view2 as reference view, predict view1's pts3d w.r.t. gt_pts2_normed_by_view2
        res1, cached_feat1 = self._forward(view2, gt_pts2_normed_by_view2, view1)
        res1['gt_pts3d'] = gt_pts1_normed_by_view2.permute(0,2,3,1)

        # recover predicted pts2 back to world scale for next auto-regression step.
        mean_v1, avg_dis_v1, mean_v2, avg_dis_v2 = stats
        res2['pts3d_w'] = res2['pts3d'].detach() * avg_dis_v1.permute(0,2,3,1) + mean_v1.permute(0,2,3,1)

        return res1, res2
    
    @torch.no_grad()
    def inference(self, view_ref, pts_ref, conf_ref, view):
        pts_ref = pts_ref.permute(0,3,1,2)
        mean = pts_ref.mean(dim=(2,3), keepdim=True)
        avg_dis = torch.norm(pts_ref - mean, dim=1, keepdim=True).mean(dim=(2,3), keepdim=True)
        pts_ref_normed = (pts_ref - mean) / (avg_dis + 1e-8)
        view_ref['conf'] = conf_ref
        res, _ = self._forward(view_ref, pts_ref_normed, view)
        res['pts3d'] = res['pts3d'] * avg_dis.permute(0,2,3,1) + mean.permute(0,2,3,1)
        return res
