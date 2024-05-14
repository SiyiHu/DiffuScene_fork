import numpy as np
import torch
from tqdm import tqdm

from scene_synthesis.datasets.threed_front_encoding import Diffusion


def unpack_data(sample, network_config, num_partial=None):
    if num_partial is None:
        num_partial = network_config["partial_num_points"]
    
    class_labels = torch.from_numpy(sample["class_labels"])[None, :num_partial, :]
    translations = torch.from_numpy(sample["translations"])[None, :num_partial, :]
    sizes = torch.from_numpy(sample["sizes"])[None, :num_partial, :]
    angles = torch.from_numpy(sample["angles"])[None, :num_partial, :]

    partial_boxes = \
        torch.cat([translations, sizes, angles, class_labels], dim=-1).contiguous()
    
    if network_config.get("objectness_dim", 0) > 0:
        objectness = torch.from_numpy(sample["objectness"])[None, :num_partial, :]
        partial_boxes = \
            torch.cat([partial_boxes, objectness], dim=-1).contiguous() 
    if network_config.get("objfeat_dim", 0) > 0:
        if network_config["objfeat_dim"] == 32:
            objfeats = torch.from_numpy(sample["objfeats_32"])[None, :num_partial, :]
        else:
            objfeats = torch.from_numpy(sample["objfeats"])[None, :num_partial, :]
        partial_boxes = \
            torch.cat([partial_boxes, objfeats], dim=-1).contiguous() 

    return partial_boxes 


def generate_layouts(network, encoded_dataset:Diffusion, num_syn_scenes, config,
                     num_known_objects=0, clip_denoised=False, sampling_rule="random", 
                     batch_size=16, device="cpu"):
    """Generate speicifed number of object layouts and also return a list of scene 
    indices corresponding to the floor plan. Each layout is a 2D array where each 
    row contain the concatenated object attributes.
    (Note: this code assumes "end" is the last object label, and, if used, 
    "start" is the second to last label.)"""
    
    # Sample floor layout
    if sampling_rule == "random":
        sampled_indices = np.random.choice(len(encoded_dataset), num_syn_scenes).tolist()
    elif sampling_rule == "uniform":
        sampled_indices = np.arange(len(encoded_dataset)).tolist() * \
            (num_syn_scenes // len(encoded_dataset))
        sampled_indices += \
            np.random.choice(len(encoded_dataset), 
                             num_syn_scenes - len(sampled_indices)).tolist()
    else:
        raise NotImplemented
    
    scene_completion = config["network"].get("room_partial_condition", False)
    if scene_completion:
        assert num_known_objects > 0
        print("Using model trained with {} existing objects for scenes "
              "completion given {} objects.".format(
                  config["network"]["partial_num_points"], num_known_objects))
    
    # Generate layouts
    network.to(device)
    network.eval()
    layout_list = []
    for i in tqdm(range(0, num_syn_scenes, batch_size)):
        scene_indices = sampled_indices[i: min(i + batch_size, num_syn_scenes)]
        
        room_mask = torch.from_numpy(np.stack([
            encoded_dataset[ind]["room_layout"] for ind in scene_indices
        ], axis=0)).to(device)
        
        if scene_completion:
            input_boxes = torch.cat([
                unpack_data(encoded_dataset[ind], config["network"], num_known_objects)
                for ind in scene_indices
            ], dim=0).to(device)
                            
            bbox_params_list = network.complete_scene(
                room_mask=room_mask,
                num_points=config["network"]["sample_num_points"],
                point_dim=config["network"]["point_dim"],
                partial_boxes=input_boxes,
                batch_size=len(scene_indices),
                out_device="cpu",
                clip_denoised=clip_denoised,
            )
        else:
            bbox_params_list = network.generate_layout(
                room_mask=room_mask,
                num_points=config["network"]["sample_num_points"],
                point_dim=config["network"]["point_dim"],
                batch_size=len(scene_indices),
                out_device="cpu",
                clip_denoised=clip_denoised,
            )
        
        for bbox_params_dict in bbox_params_list:
            boxes = encoded_dataset.post_process(bbox_params_dict)
            bbox_params = {k: v.numpy()[0] for k, v in boxes.items()}
            layout_list.append(bbox_params)
    
    return sampled_indices, layout_list
