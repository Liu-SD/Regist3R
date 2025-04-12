# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# utilities needed for the inference
# --------------------------------------------------------
import tqdm
import numpy as np
import torch
from dust3r.utils.device import to_cpu, collate_with_cat
from dust3r.inference import loss_of_one_batch as loss_of_one_batch_dust3r
from regist3r.utils import build_tree


def loss_of_one_step(batch, model, criterion, device, ret=None):
    view1, view2 = batch
    ignore_keys = set(['depthmap', 'dataset', 'label', 'instance', 'idx', 'true_shape', 'rng'])
    for view in [view1, view2]:
        for name in view.keys():  # pseudo_focal
            if name in ignore_keys:
                continue
            view[name] = view[name].to(device, non_blocking=True)
    
    if 'pr_pts3d' not in view1: # first step
        view1['pr_pts3d'] = view1['pts3d']
        view1['conf'] = torch.ones_like(view1['img'][:,0])
    pred1, pred2 = model(view1, view2)
    # prepare inputs for next step
    view2['pr_pts3d'] = pred2['pts3d_w']
    conf = pred2['conf'].detach()
    view2['conf'] = conf / (1 + conf) # exp to sigmoid

    loss = criterion(view1, view2, pred1, pred2) if criterion is not None else None

    result = dict(view1=view1, view2=view2, pred1=pred1, pred2=pred2, loss=loss)
    return result[ret] if ret else result

@torch.no_grad()
def inference(sequence, filelist, dust3r_model, regist3r_model, mast3r_model, retrieval_model, device, 
              tree_type='MST', start=-1, tree_compression_factor=0, affinity_mode='asmk', verbose=True):

    def to_device(view):
        ignore_keys = set(['depthmap', 'dataset', 'label', 'instance', 'idx', 'true_shape', 'rng'])
        for name in view.keys():  # pseudo_focal
            if name in ignore_keys:
                continue
            view[name] = view[name].to(device, non_blocking=True)
        return view
        
    if verbose:
        print(f'>> Inference with model on image sequence of length {len(sequence)}')
    result = []

    if affinity_mode == 'asmk':
        affinity_matrix = get_affinity_matrix_asmk(filelist, retrieval_model, backnone=mast3r_model, device=device)
    else:
        pairwise = True if affinity_mode == 'pairwise' else False
        affinity_matrix = get_affinity_matrix(to_device(collate_with_cat(sequence)), dust3r_model, pairwise=pairwise)
    if start == -1:
        start = affinity_matrix.sum(dim=0).argmax().item()
    tree = build_tree(affinity_matrix, tree_type, start, tree_compression_factor)
    if verbose:
        print('\n', tree)

    init_pair = [(sequence[tree.value], sequence[tree.childs[0].value])]
    # init_pair = [(sequence[start], sequence[start])]
    dust3r_res = loss_of_one_batch_dust3r(collate_with_cat(init_pair), dust3r_model, None, device)
    view1, view2, pred1, pred2 = [dust3r_res[k] for k in 'view1 view2 pred1 pred2'.split()]
    result.append({'view': view1, 'pred': pred1, 'index': [start], 'ref_index': [start]})

    view_nodes = tree.childs
    tree.pts = pred1['pts3d']
    tree.conf = torch.ones_like(pred1['pts3d'][...,0])
    depth = 1
    batch_size = 64
    while len(view_nodes) > 0:
        next_view_nodes = []
        for i in range(0, len(view_nodes), batch_size):
            batch_view_nodes = view_nodes[i:i+batch_size]
            parent_views = to_device(collate_with_cat([sequence[v.parent.value] for v in batch_view_nodes]))
            views = to_device(collate_with_cat([sequence[v.value] for v in batch_view_nodes]))
            parent_pts = torch.cat([v.parent.pts for v in batch_view_nodes])
            parent_confs = torch.cat([v.parent.conf for v in batch_view_nodes])
            parent_indices = [v.parent.value for v in batch_view_nodes]
            view_indices = [v.value for v in batch_view_nodes]
            if verbose:
                total_done = sum([len(r['index']) for r in result])
                print(f"({total_done}/{len(sequence)}|depth: {depth}) inference between {parent_indices} and {view_indices}.")
            res = regist3r_model.inference(parent_views, parent_pts, parent_confs, views)
            # register with dust3r in Light3R-SfM mode
            # res = infer_then_align(dust3r_model, parent_views, parent_pts, views, device=device)
            result.append({'view': views, 'pred': res, 'index': view_indices, 'ref_index': parent_indices})
            for i, view_node in enumerate(batch_view_nodes):
                view_node.pts = res['pts3d'][i:i+1]
                view_node.conf = res['conf'][i:i+1] / (res['conf'][i:i+1]+1) # log-sigmoid
                # Ablation study on confidence-aware autoregressive training.
                # view_node.conf = torch.ones_like(res['conf'][i:i+1])
                next_view_nodes.extend(view_node.childs)
        view_nodes = next_view_nodes
        depth += 1
    result = collate_with_cat(result, lists=False)
    result = sort_by(result, np.argsort(result["index"]).tolist())
    return result

def sort_by(whatever, index):
    if isinstance(whatever, dict):
        return {k: sort_by(v, index) for k, v in whatever.items()}
    if isinstance(whatever, torch.Tensor) or isinstance(whatever, np.ndarray):
        return whatever[index]
    if isinstance(whatever, list):
        return [whatever[i] for i in index]
    else:
        return whatever

def infer_then_align(dust3r_model, view_ref, pts_ref, view, device):
    pair = [(view_ref, view)]
    dust3r_res = loss_of_one_batch_dust3r(collate_with_cat(pair), dust3r_model, None, device)
    view1, view2, pred1, pred2 = [dust3r_res[k] for k in 'view1 view2 pred1 pred2'.split()]
    from regist3r.utils import procrustes_alignment
    s, R, t = procrustes_alignment(pred1['pts3d'].view(-1, 3), pts_ref.view(-1, 3), pred1['conf'].view(-1))
    view_pts3d = s * torch.einsum('ij, bhwi -> bhwj', R, pred2['pts3d_in_other_view']) + t
    return {'pts3d': view_pts3d, 'conf': pred2['conf']}

@torch.no_grad()
def get_affinity_matrix_asmk(filelist, retrieval_model, backnone, device):
    from mast3r.retrieval.processor import Retriever
    retriever = Retriever(retrieval_model, backbone=backnone, device=device)
    affinity_matrix = retriever(filelist)
    affinity_matrix = torch.from_numpy(affinity_matrix).to(device)
    affinity_matrix.masked_fill_(torch.eye(len(filelist), device=device, dtype=torch.bool), 1.)
    return affinity_matrix

@torch.no_grad()
@torch.amp.autocast('cuda')
def get_affinity_matrix(views, dust3r_model, pairwise=False):
    imgs = views['img']
    B = imgs.shape[0]
    # Recover true_shape when available, otherwise assume that the img shape is the true one
    shapes = views.get('true_shape', torch.tensor(imgs.shape[-2:])[None].repeat(B, 1))

    batch_size = 16
    feats = []
    poss = []
    patch_size = dust3r_model.patch_embed.patch_size[0]
    bar = tqdm.tqdm(total=B, desc="Building affinity matrix")
    for img, shape in zip(imgs.split(batch_size), shapes.split(batch_size)):
        feat, pos, _ = dust3r_model._encode_image(img, shape)

        B, _, H, W = img.shape
        if not pairwise:
            nH, nW = H//patch_size, W//patch_size
            feat = feat.view(B, nH, nW, -1)
            feat = torch.nn.functional.normalize(torch.cat((
                feat[:, :nH//2, :nW//2].mean(dim=(1,2)), 
                feat[:, :nH//2, nW//2:].mean(dim=(1,2)),
                feat[:, nH//2:, :nW//2].mean(dim=(1,2)), 
                feat[:, nH//2:, nW//2:].mean(dim=(1,2)),
            ), dim=-1), dim=1)
            # feat = torch.nn.functional.normalize(feat.mean(dim=1), dim=1)

        bar.update(B)
        feats.append(feat)
        poss.append(pos)
    bar.close()

    affinity_matrix = torch.zeros(imgs.shape[0], imgs.shape[0], device=imgs.device)

    if not pairwise:
        for i in range(len(feats)):
            for j in range(i+1):
                feat1, feat2 = feats[i], feats[j] # [B,C]
                feat1 = feat1.unsqueeze(0)
                feat2 = feat2.unsqueeze(1)
                sub_matrix = (feat1 * feat2).sum(dim=-1)
                affinity_matrix[j*batch_size:(j+1)*batch_size, i*batch_size:(i+1)*batch_size] = sub_matrix
                affinity_matrix[i*batch_size:(i+1)*batch_size, j*batch_size:(j+1)*batch_size] = sub_matrix.transpose(0,1)
    else:
        feats = torch.cat(feats, dim=0)
        poss = torch.cat(poss, dim=0)
        from itertools import combinations
        all_pairs_i, all_pairs_j = list(zip(*combinations(range(feats.shape[0]), 2)))
        all_pairs_i, all_pairs_j = list(all_pairs_i), list(all_pairs_j)
        bar = tqdm.tqdm(total=len(all_pairs_i), desc="Compute pairwise confidence")
        for k in range(0, len(all_pairs_i), batch_size):
            pairs_i, pairs_j = all_pairs_i[k:k+batch_size], all_pairs_j[k:k+batch_size]
            feat_i, feat_j = feats[pairs_i], feats[pairs_j]
            pos_i, pos_j = poss[pairs_i], poss[pairs_j]
            shape_i, shape_j = shapes[pairs_i], shapes[pairs_j]
            dec_i, dec_j = dust3r_model._decoder(feat_i, pos_i, feat_j, pos_j)
            conf_i = dust3r_model._downstream_head(1, dec_i, shape_i)['conf'].mean(dim=(1,2))
            conf_j = dust3r_model._downstream_head(2, dec_j, shape_j)['conf'].mean(dim=(1,2))
            conf = (conf_i + conf_j) / 2
            conf = conf / (conf+1) # exp to sigmoid
            affinity_matrix[pairs_i, pairs_j] = conf
            affinity_matrix[pairs_j, pairs_i] = conf

            bar.update(batch_size)
        bar.close()
        print(affinity_matrix)

    return affinity_matrix
