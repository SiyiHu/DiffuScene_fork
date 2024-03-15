import numpy as np
import torch
from tqdm import tqdm

from scene_synthesis.datasets.threed_front_encoding import Diffusion


def generate_layouts(network, encoded_dataset:Diffusion, num_syn_scenes, config,
                     clip_denoised, sampling_rule="random", batch_size=16, 
                     device="cpu"):
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
        
    # Generate layouts
    network.to(device)
    network.eval()
    layout_list = []
    for i in tqdm(range(0, num_syn_scenes, batch_size)):
        scene_indices = sampled_indices[i: min(i + batch_size, num_syn_scenes)]
        
        room_mask = torch.from_numpy(np.stack([
            encoded_dataset[ind]["room_layout"] for ind in scene_indices
        ], axis=0)).to(device)
        
        bbox_params_list = network.generate_layout(
            room_mask=room_mask.to(device),
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
