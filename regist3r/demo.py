# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# gradio demo
# --------------------------------------------------------
import argparse
import gradio
import os
import functools
import copy

from regist3r.inference import inference
from dust3r.utils.image import load_images, rgb, colorize_depth
from dust3r.utils.device import to_numpy
from dust3r.demo import get_3D_model_from_scene
from regist3r.cloud_opt import global_aligner

import matplotlib.pyplot as pl


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser_url = parser.add_mutually_exclusive_group()
    parser_url.add_argument("--local_network", action='store_true', default=False,
                            help="make app accessible on local network: address will be set to 0.0.0.0")
    parser_url.add_argument("--server_name", type=str, default=None, help="server url, default is 127.0.0.1")
    parser.add_argument("--image_size", type=int, default=512, choices=[512, 224], help="image size")
    parser.add_argument("--server_port", type=int, help=("will start gradio app on this port (if available). "
                                                         "If None, will search for an available port starting at 7860."),
                        default=None)
    parser.add_argument("--dust3r_weights", type=str, help="path to the dust3r weights", default=None)
    parser.add_argument("--regist3r_weights", type=str, help="path to the regist3r weights", default=None)
    parser.add_argument("--mast3r_weights", type=str, help="path to the mast3r weights", default=None)
    parser.add_argument("--retrieval_weights", type=str, help="path to the retrieval_model weights", default=None)
    parser.add_argument("--device", type=str, default='cuda', help="pytorch device")
    parser.add_argument("--tmp_dir", type=str, default=None, help="value for tempfile.tempdir")
    parser.add_argument("--silent", action='store_true', default=False,
                        help="silence logs")
    return parser


def get_reconstructed_scene(outdir, dust3r_model, regist3r_model, mast3r_model, retrieval_model, device, 
                            silent, image_size, filelist,  tree_type, start_frame, tree_compression_factor, 
                            affinity_mode, min_conf_thr, as_pointcloud, mask_sky, clean_depth, 
                            transparent_cams, cam_size):
    """
    from a list of images, run dust3r inference, global aligner.
    then run get_3D_model_from_scene
    """
    imgs = load_images(filelist, size=image_size, verbose=not silent, consistent_shape=True)
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1

    output = inference(imgs, filelist, dust3r_model, regist3r_model, mast3r_model, retrieval_model, device, 
                       affinity_mode=affinity_mode,
                       tree_type=tree_type, start=start_frame, 
                       tree_compression_factor=tree_compression_factor, 
                       verbose=not silent)

    scene = global_aligner(output, device=device)

    outfile = get_3D_model_from_scene(outdir, silent, scene, min_conf_thr, as_pointcloud, mask_sky,
                                      clean_depth, transparent_cams, cam_size)

    # also return rgb, depth and confidence imgs
    # depth is normalized with the max value for all images
    # we apply the jet colormap on the confidence maps
    rgbimg = scene.imgs
    depths = to_numpy(scene.get_depthmaps())
    confs = to_numpy([c for c in scene.im_conf_display])
    cmap = pl.get_cmap('jet')
    depths_max = max([d.max() for d in depths])
    depths = [d / depths_max for d in depths]
    confs_max = max([d.max() for d in confs])
    confs = [cmap(d / confs_max) for d in confs]

    imgs = []
    for i in range(len(rgbimg)):
        ref_i = scene.indices.index(scene.ref_indices[i])
        imgs.append(rgbimg[ref_i])
        imgs.append(rgbimg[i])
        imgs.append(colorize_depth(depths[i]))
        imgs.append(rgb(confs[i]))

    return scene, outfile, imgs

def set_start_frame_options(inputfiles, start_frame):
    if inputfiles is None:
        inputfiles = []
    start_frame = gradio.Dropdown([('auto', -1)] + [(os.path.basename(f), i) for i, f in enumerate(inputfiles)],
                                  value=-1, label='Start Frame',
                                  info='Select a frame as the root to build shortest-path-tree',
                                  interactive=True)
    return start_frame


def main_demo(tmpdirname, dust3r_model, regist3r_model, mast3r_model, retrieval_model, device, image_size, server_name, server_port, silent=False):
    recon_fun = functools.partial(get_reconstructed_scene, tmpdirname, dust3r_model, regist3r_model, mast3r_model, retrieval_model, device, silent, image_size)
    model_from_scene_fun = functools.partial(get_3D_model_from_scene, tmpdirname, silent)
    with gradio.Blocks(css=""".gradio-container {margin: 0 !important; min-width: 100%};""", title="Regist3R Demo") as demo:
        # scene state is save so that you can change conf_thr, cam_size... without rerunning the inference
        scene = gradio.State(None)
        gradio.HTML('<h2 style="text-align: center;">Regist3R Demo</h2>')
        with gradio.Column():
            inputfiles = gradio.File(file_count="multiple")
            with gradio.Row():
                start_frame = gradio.Dropdown([("auto", -1)],
                                              value=-1, label="Start Frame",
                                              info="Select a frame as the root to build shortest-path-tree",
                                              interactive=True)
                affinity_mode = gradio.Dropdown([("ASMK: Aggregated Selective Match Kernels", "asmk"),
                                                 ("Pairwise: compute pairwise confidence with Dust3R decoder", 'pairwise'),
                                                 ("Normalized Distance: similarity of normalized features of Dust3R encoder", 'normdist')],
                                                 label="Affinity matrix",
                                                 value='asmk', info="Metric to build affinity matrix", interactive=True)
                tree_type = gradio.Dropdown([("SPT: shortest path tree", "SPT"),
                                             ("MST: minimal spanning tree", "MST")],
                                             value="MST", label="Tree Type",
                                             info="Type of tree for incremental registerization",
                                             interactive=True)
                tree_compression_factor = gradio.Dropdown([(f"{2**i}", i) for i in range(6)],
                                                          value=0, label="Tree Compression Factor",
                                                          info="Compress tree depth. Set this when MST tree is too deep.",
                                                          interactive=True)

            run_btn = gradio.Button("Run")

            with gradio.Row():
                # adjust the confidence threshold
                min_conf_thr = gradio.Slider(label="min_conf_thr", value=3.0, minimum=1.0, maximum=20, step=0.1)
                # adjust the camera size in the output pointcloud
                cam_size = gradio.Slider(label="cam_size", value=0.05, minimum=0.001, maximum=0.1, step=0.001)
            with gradio.Row():
                as_pointcloud = gradio.Checkbox(value=False, label="As pointcloud")
                # two post process implemented
                mask_sky = gradio.Checkbox(value=False, label="Mask sky")
                clean_depth = gradio.Checkbox(value=True, label="Clean-up depthmaps")
                transparent_cams = gradio.Checkbox(value=False, label="Transparent cameras")

            outmodel = gradio.Model3D()
            outgallery = gradio.Gallery(label='ref_rgb,rgb,depth,confidence', columns=4, height="100%")

            # events
            inputfiles.change(set_start_frame_options,
                              inputs=[inputfiles, start_frame],
                              outputs=[start_frame])
            run_btn.click(fn=recon_fun,
                          inputs=[inputfiles, tree_type, start_frame, tree_compression_factor, 
                                  affinity_mode, min_conf_thr, as_pointcloud, mask_sky, clean_depth, 
                                  transparent_cams, cam_size],
                          outputs=[scene, outmodel, outgallery])
            min_conf_thr.release(fn=model_from_scene_fun,
                                 inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                         clean_depth, transparent_cams, cam_size],
                                 outputs=outmodel)
            cam_size.change(fn=model_from_scene_fun,
                            inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                    clean_depth, transparent_cams, cam_size],
                            outputs=outmodel)
            as_pointcloud.change(fn=model_from_scene_fun,
                                 inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                         clean_depth, transparent_cams, cam_size],
                                 outputs=outmodel)
            mask_sky.change(fn=model_from_scene_fun,
                            inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                    clean_depth, transparent_cams, cam_size],
                            outputs=outmodel)
            clean_depth.change(fn=model_from_scene_fun,
                               inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                       clean_depth, transparent_cams, cam_size],
                               outputs=outmodel)
            transparent_cams.change(model_from_scene_fun,
                                    inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                            clean_depth, transparent_cams, cam_size],
                                    outputs=outmodel)
    demo.launch(share=False, server_name=server_name, server_port=server_port)
