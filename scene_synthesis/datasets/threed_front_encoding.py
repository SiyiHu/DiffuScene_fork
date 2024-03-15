# 
# Modified get_dataset_raw_and_encoded() and get_encoded_dataset() to use
# Threed_Front library
# 

from threed_front.datasets.threed_front_encoding_base import *
from threed_front.datasets import get_raw_dataset
from .threed_front_dataset import ObjFeatEncoder, ObjFeat32Encoder, \
    Scale_CosinAngle, Scale_CosinAngle_ObjfeatsNorm, Add_Text, Diffusion, AutoregressiveWOCM


def get_dataset_raw_and_encoded(
    config,
    filter_fn=lambda s: s,
    path_to_bounds=None,
    augmentations=None,
    split=["train", "val"],
    max_length=None,
    with_room_layout=True,
):
    dataset = get_raw_dataset(
        config, filter_fn, path_to_bounds, split, 
        include_room_mask=with_room_layout
    )
    encoding = dataset_encoding_factory(
        config.get("encoding_type"),
        dataset,
        augmentations,
        config.get("box_ordering", None),
        max_length
    )

    return dataset, encoding


def get_encoded_dataset(
    config,
    filter_fn=lambda s: s,
    path_to_bounds=None,
    augmentations=None,
    split=["train", "val"],
    max_length=None,
    with_room_layout=True
):
    _, encoding = get_dataset_raw_and_encoded(
        config, filter_fn, path_to_bounds, augmentations, split, max_length, 
        with_room_layout
    )
    return encoding

def dataset_encoding_factory(
    name,
    dataset,
    augmentations=None,
    box_ordering=None,
    max_length=None,
):
    # list of object features
    feature_keys = ["class_labels", "translations", "sizes", "angles"]
    if "objfeats" in name:
        if "lat32" in name:
            feature_keys.append("objfeats_32")
            print("use lat32 as objfeats")
        else:
            feature_keys.append("objfeats")
            print("use lat64 as objfeats")
    
    # NOTE: The ordering might change after augmentations so really it should
    #       be done after the augmentations. For class frequencies it is fine
    #       though.
    if "cached" in name:
        dataset_collection = CachedDatasetCollection(dataset)
        if box_ordering:
            dataset_collection = \
                OrderedDataset(dataset_collection, feature_keys, box_ordering)
    else:
        box_ordered_dataset = BoxOrderedDataset(dataset, box_ordering)

        class_labels = ClassLabelsEncoder(box_ordered_dataset)
        translations = TranslationEncoder(box_ordered_dataset)
        sizes = SizeEncoder(box_ordered_dataset)
        angles = AngleEncoder(box_ordered_dataset)
        objfeats = ObjFeatEncoder(box_ordered_dataset)
        objfeats_32 = ObjFeat32Encoder(box_ordered_dataset)

        if name == "basic":
            return DatasetCollection(
                class_labels,
                translations,
                sizes,
                angles,
                objfeats,
                objfeats_32
            )
        
        room_layout = RoomLayoutEncoder(box_ordered_dataset)
        dataset_collection = DatasetCollection(
            room_layout,
            class_labels,
            translations,
            sizes,
            angles,
            objfeats,
            objfeats_32
        )

    if isinstance(augmentations, list):
        for aug_type in augmentations:
            if aug_type == "rotations":
                print("Applying rotation augmentations")
                dataset_collection = RotationAugmentation(dataset_collection)
            elif aug_type == "fixed_rotations":
                print("Applying fixed rotation augmentations")
                dataset_collection = RotationAugmentation(dataset_collection, fixed=True)
            elif aug_type == "jitter":
                print("Applying jittering augmentations")
                dataset_collection = Jitter(dataset_collection)

    if "textfix" in name:
        print("add text into input dict for evalation")
        dataset_collection = Add_Text(dataset_collection, eval=True)
    elif "text" in name:
        print("add text into input dict for training")
        dataset_collection = Add_Text(dataset_collection, eval=False)
        

    # Scale the input
    if "cosin_angle" in name and "objfeatsnorm" in name:
        print('use consin_angles instead of original angles, AND use normalized objfeats')
        dataset_collection = Scale_CosinAngle_ObjfeatsNorm(dataset_collection)
    elif "cosin_angle" in name:
        print('use consin_angles instead of original angles')
        dataset_collection = Scale_CosinAngle(dataset_collection)
    else:
        dataset_collection = Scale(dataset_collection)

    # for diffusion (represent objectness as the last channel of class label)
    if "diffusion" in name:
        if "eval" in name:
            return dataset_collection
        elif "wocm_no_prm" in name:
            return Diffusion(dataset_collection)
        elif "wocm" in name:
            dataset_collection = Permutation(dataset_collection, feature_keys)
            return Diffusion(dataset_collection)
        
    # for autoregressive model
    elif "autoregressive" in name:
        if "eval" in name:
            return dataset_collection
        elif "wocm_no_prm" in name:
            return AutoregressiveWOCM(dataset_collection)
        elif "wocm" in name:
            dataset_collection = Permutation(dataset_collection, feature_keys)
            return AutoregressiveWOCM(dataset_collection)
    else:
        raise NotImplementedError()