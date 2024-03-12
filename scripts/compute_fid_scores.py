# 
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler
# 

"""Script for computing the FID score between real and synthesized scenes.
"""
import argparse
import os
import torch
from PIL import Image
from cleanfid import fid
import shutil

from scene_synthesis.datasets.splits_builder import CSVSplitsBuilder
from scene_synthesis.datasets.threed_front import CachedThreedFront


class ThreedFrontRenderDataset(object):
    def __init__(self, dataset, 
                 image_name="rendered_scene_notexture_256_nofloor.png"):
        self.dataset = dataset
        self.image_name = image_name
        print("Use {} as rendered scene image.".format(image_name))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image_path = os.path.join(
            os.path.dirname(self.dataset[idx].image_path),
            self.image_name
        )
        img = Image.open(image_path)
        return img


def main(argv):
    parser = argparse.ArgumentParser(
        description=("Compute the FID scores between the real and the "
                     "synthetic images")
    )
    parser.add_argument(
        "path_to_real_renderings",
        help="Path to the folder containing the real renderings"
    )
    parser.add_argument(
        "path_to_synthesized_renderings",
        help="Path to the folder containing the synthesized"
    )
    parser.add_argument(
        "--path_to_annotations",
        default=None,
        help="Path to the file containing dataset splits"
        "(default:../config/<room_type>_threed_front_splits.csv)"
    )
    parser.add_argument(
        "--compare_all",
        action="store_true",
        help="if compare all"
    )
    parser.add_argument(
        "--compare_test",
        action="store_true",
        help="if compare with the test split only"
    )

    args = parser.parse_args(argv)
    # Set default path_to_annotations if not specified
    if args.path_to_annotations is None:
        for room_type in ["bedroom", "diningroom", "livingroom"]:
            default_path = f"../config/{room_type}_threed_front_splits.csv"
            if room_type in args.path_to_real_renderings and os.path.exists(default_path):
                args.path_to_annotations = default_path
                break
        else:
            print("Cannot find dataset splits automatically. "
                  "Please use '--path_to_annotations' to provide file path.")
            return

    # Create Real datasets
    config = dict(
        train_stats="dataset_stats.txt",
        room_layout_size="256,256"
    )
    splits_builder = CSVSplitsBuilder(args.path_to_annotations)
    if args.compare_all:
        scene_ids = splits_builder.get_splits(["train", "val", "test"])
    elif args.compare_test:
        scene_ids = splits_builder.get_splits(["test"])
    else:
        scene_ids = splits_builder.get_splits(["train", "val"])
    test_real = ThreedFrontRenderDataset(CachedThreedFront(
        args.path_to_real_renderings, config=config, scene_ids=scene_ids
    ))

    path_to_test_real = "../output/fid_tmp/test_real/"
    print("Generating a temporary folder with test_real images:", path_to_test_real)
    os.makedirs(path_to_test_real)
    for i, di in enumerate(test_real):
        di.save("{}/{:05d}.png".format(path_to_test_real, i))
    print('number of synthesized images :', len(test_real))

    path_to_test_fake = "../output/fid_tmp/test_fake/"
    print("Generating a temporary folder with test_fake images:", path_to_test_fake)
    os.makedirs(path_to_test_fake)
    synthesized_images = [
        os.path.join(args.path_to_synthesized_renderings, oi)
        for oi in os.listdir(args.path_to_synthesized_renderings)
        if oi.endswith(".png")
    ]
    print('number of synthesized images :', len(synthesized_images))

    for i, fi in enumerate(synthesized_images):
        shutil.copyfile(fi, "{}/{:05d}.png".format(path_to_test_fake, i))

    # Compute the FID score
    fid_score = fid.compute_fid(path_to_test_real, path_to_test_fake, device=torch.device("cpu"))
    print('fid score:', fid_score)
    kid_score = fid.compute_kid(path_to_test_real, path_to_test_fake, device=torch.device("cpu"))
    print('kid score:', kid_score)
    
    input("FID/KID score computed. Press any key to remove temporary image directories...")
    os.system('rm -r %s'%path_to_test_real)
    os.system('rm -r %s'%path_to_test_fake)


if __name__ == "__main__":
    main(None)

# python compute_fid_scores.py /cluster/balrog/jtang/3d_front_processed/bedrooms_notexture_nofloor_whiteground/ /cluster/balrog/jtang/ATISS_exps/diffusion_bedrooms_objfeats_lat32_v/gen_clip_24000/ ../config/bedroom_threed_front_splits.csv
# python compute_fid_scores.py /cluster/balrog/jtang/3d_front_processed/livingrooms_notexture_nofloor_whiteground/ /cluster/balrog/jtang/ATISS_exps/diffusion_livingrooms_permaug_fixedrotaug_unet1d_dim512_nomask_instancond_cosinangle_ddpm_separateclsbbox/gen_top2down_notexture_nofloor-58000/ ../config/livingroom_threed_front_splits.csv 
# python compute_fid_scores.py /cluster/balrog/jtang/3d_front_processed/diningrooms_notexture_nofloor_whiteground/ /cluster/balrog/jtang/ATISS_exps/diffusion_diningrooms_permaug_fixedrotaug_unet1d_dim512_nomask_instancond_cosinangle_ddpm_separateclsbbox_modinstan_objfeats_biou/gen_top2down_notexture_nofloor_retrifeats_combsize-59800/ ../config/diningroom_threed_front_splits.csv 