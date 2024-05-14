"""Script used for generating results using a previously trained model."""
import argparse
import logging
import os
import sys
import shutil
import pickle

import numpy as np
import torch

from training_utils import load_config

from threed_front.datasets import filter_function, get_raw_dataset
from threed_front.evaluation import ThreedFrontResults

from scene_synthesis.datasets.threed_front_encoding import get_dataset_raw_and_encoded
from scene_synthesis.networks import build_network
from scene_synthesis.evaluation.utils import generate_layouts


def main(argv):
    parser = argparse.ArgumentParser(
        description="Generate scenes using a previously trained model"
    )

    parser.add_argument(
        "weight_file",
        help="Path to a pretrained model"
    )
    parser.add_argument(
        "--output_directory",
        default="./tmp",
        help="Path to the output directory"
    )
    parser.add_argument(
        "--config_file",
        default=None,
        help="Path to the file that contains the experiment configuration"
        "(default: config.yaml in the model directory)"
    )
    parser.add_argument(
        "--n_known_objects",
        default=0,
        type=int,
        help="Number of existing objects for scene completion task"
    )
    parser.add_argument(
        "--clip_denoised",
        action="store_true",
        help="if clip_denoised"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the sampling floor plan"
    )
    parser.add_argument(
        "--n_syn_scenes",
        default=1000,
        type=int,
        help="Number of scenes to be synthesized"
    )
    parser.add_argument(
        "--batch_size",
        default=128,
        type=int,
        help="Number of synthesized scene in each batch"
    )
    parser.add_argument(
        "--result_tag",
        default=None,
        help="Save results to a sub-directory if result_tag is provided"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU ID"
    )

    args = parser.parse_args(argv)

    # Disable trimesh's logger
    logging.getLogger("trimesh").setLevel(logging.ERROR)

    # Set the random seed
    np.random.seed(args.seed)
    torch.manual_seed(np.random.randint(np.iinfo(np.int32).max))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(np.random.randint(np.iinfo(np.int32).max))

    if args.gpu < torch.cuda.device_count():
        device = torch.device("cuda:{}".format(args.gpu))
    else:
        device = torch.device("cpu")
    print("Running code on", device)

    # Check if result_dir exists and if it doesn't create it
    if args.result_tag is None:
        result_dir = args.output_directory
    else:
        result_dir = os.path.join(args.output_directory, args.result_tag)
    if os.path.exists(result_dir) and \
        len(os.listdir(result_dir)) > 0:
        input("{} direcotry is non-empty. Press any key to remove all files..." \
              .format(result_dir))
        for fi in os.listdir(result_dir):
            os.remove(os.path.join(result_dir, fi))
    else:
        os.makedirs(result_dir, exist_ok=True)

    # Run control files to save
    path_to_config = os.path.join(result_dir, "config.yaml")
    path_to_results = os.path.join(result_dir, "results.pkl")

    # Parse the config file
    if args.config_file is None:
        args.config_file = os.path.join(os.path.dirname(args.weight_file), "config.yaml")
    config = load_config(args.config_file)
    # if "_eval" not in config["data"]["encoding_type"]:
    #     config["data"]["encoding_type"] += "_eval"
    if not os.path.exists(path_to_config) or \
        not os.path.samefile(args.config_file, path_to_config):
        shutil.copyfile(args.config_file, path_to_config)

    # Raw training data (for record keeping)
    raw_train_dataset = get_raw_dataset(
        config["data"], 
        filter_fn=filter_function(
            config["data"],
            split=config["training"].get("splits", ["train", "val"])
        ),
        split=config["training"].get("splits", ["train", "val"]),
        include_room_mask=True
    ) 

    # Get Scaled dataset encoding (without data augmentation)
    raw_dataset, encoded_dataset = get_dataset_raw_and_encoded(
        config["data"],
        filter_fn=filter_function(
            config["data"],
            split=config["validation"].get("splits", ["test"])
        ),
        split=config["validation"].get("splits", ["test"]),
    )
    print("Loaded {} scenes with {} object types ({} labels):".format(
        len(encoded_dataset), encoded_dataset.n_object_types, encoded_dataset.n_classes))
    print(encoded_dataset.class_labels)    

    # Build network with saved weights
    network, _, _ = build_network(
        encoded_dataset.feature_size, encoded_dataset.n_classes,
        config, args.weight_file, device=device
    )
    network.eval()

    # Generate final results
    sampled_indices, layout_list = generate_layouts(
        network, encoded_dataset, args.n_syn_scenes, config, args.n_known_objects,
        args.clip_denoised, "random", args.batch_size, device
    )
    
    threed_front_results = ThreedFrontResults(
        raw_train_dataset, raw_dataset, config, sampled_indices, layout_list
    )
    
    pickle.dump(threed_front_results, open(path_to_results, "wb"))
    print("Saved result to:", path_to_results)
    
    kl_divergence = threed_front_results.kl_divergence()
    print("object category kl divergence:", kl_divergence)
           

if __name__ == "__main__":
    main(sys.argv[1:])