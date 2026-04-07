import brainio_collection
import numpy as np
import LIB_dicarlo_analysis as lib
from brainscore_vision.benchmark_helpers.neural_common import average_repetition
import xarray as xr
import pickle
import os
from PIL import Image

# PARAMETERS
region = "V4"  # "IT" or "V4"
dataset_location = "./"

# PUBLIC DATASET SETTINGS
dataset = "dicarlo.MajajHong2015.public"
images_folder_path = "path/to/.brainio/image_dicarlo_hvm-public/"  # Folder containing the public MajajHong2015 image files.
output_image_dataset_name = "DicarloAsGenai_public"
output_representations_dataset_name = f"representations_DicarloAsGenai_public_{region}"

os.makedirs(os.path.join(dataset_location, output_image_dataset_name), exist_ok=True)
os.makedirs(os.path.join(dataset_location, output_representations_dataset_name), exist_ok=True)

# LOAD THE DATASET
responses = brainio_collection.get_assembly(dataset)

# <editor-fold desc="Convert to python dictionary. Output: responses_dictionary">
# RETRIEVE NAMES OF COORDINATE INDICES
# Retrieve names of the levels of the 'neuroid' coordinate
neuroid_index_names = responses.coords['neuroid'].to_index().names
# Retrieve names of the levels of the 'presentation' coordinate
presentation_index_names = responses.coords['presentation'].to_index().names

# CREATE DICTIONARY
# (create numpy dictionary from data, which I am more familiar handling)
responses_dictionary = {
    "values": np.array(responses.values)[:, :, 0],  # remove the dummy time dimension
    "neuroid": {},
    "presentation": {},
    "coordinate_positions": ["neuroid", "presentation"],
    # this is to keep track to which index of the values array corresponds which coordinate
}
# print("loaded responses values")
# print(type(responses_dictionary["values"]))
# print(responses_dictionary["values"].shape)
# print(responses_dictionary["values"])

# fill the dictionary
for index in neuroid_index_names:
    responses_dictionary["neuroid"][index] = np.array(responses.neuroid[index].values)
    # print("loaded neuroid index: " + index)
    # print(type(responses_dictionary["neuroid"][index]))
    # print(responses_dictionary["neuroid"][index].shape)
    # print(responses_dictionary["neuroid"][index])

for index in presentation_index_names:
    responses_dictionary["presentation"][index] = np.array(responses.presentation[index].values)
    # print("loaded presentation index: " + index)
    # print(type(responses_dictionary["presentation"][index]))
    # print(responses_dictionary["presentation"][index].shape)
    # print(responses_dictionary["presentation"][index])
# </editor-fold>

# AVERAGE OVER REPETITIONS OF THE SAME IMAGE
responses_avg = lib.group_average(responses_dictionary, "presentation", "image_id", over=["repetition"])

# EXTRACT NEURONS FROM THE SPECIFIED REGION (V4 or IT)
print(f"\nCreating dataset for neurons in region: {region}")
# retrieve indices where the region is the desired one
region_indices = np.where(responses_avg["neuroid"]["region"] == region)[0]
# slice the dataset over those indices
responses_region = lib.select(responses_avg, "neuroid", region_indices)
# Also keep the non-averaged data for later grouping by repetition
responses_nonavg_region = lib.select(responses_dictionary, "neuroid", region_indices)

# EXTRACT CATEGORIES AND CREATE CATEGORY MAPPING
categories = np.unique(responses_region["presentation"]["category_name"])
# Sort categories alphabetically.
categories = sorted(categories)
# Create a dictionary mapping category names to indices.
cat_to_idx = {cat: idx for idx, cat in enumerate(categories)}

# Save the dictionary as categories.pkl in the dataset_dir.
output_path = os.path.join(dataset_location, output_image_dataset_name, "categories.pkl")
with open(output_path, "wb") as f:
    pickle.dump(cat_to_idx, f)
print(f"Saved category mapping for {len(categories)} categories to {output_path}")

for category in categories:

    # CREATE FOLDERS
    os.makedirs(os.path.join(dataset_location, output_image_dataset_name, category, "images"), exist_ok=True)
    os.makedirs(os.path.join(dataset_location, output_image_dataset_name, category, "bboxes"), exist_ok=True)
    os.makedirs(os.path.join(dataset_location, output_representations_dataset_name, category), exist_ok=True)

    # CREATE THE METADATA DICTIONARY
    presentation = {
        "img_relative_path": [],  # has the form: os.path.join(category, "images", f"{image_name}.jpg")
        "category": [],
        "category_idx": [],
        # "category_predicted": [],  # absent for brain data
        # "category_idx_predicted": [],  # absent for brain data
        "bbox_center_x": [],  # Note: these are not normalized by training labels mean and variance in this case
        # (but are normalized by pixels). It does not matter much anyway
        "bbox_center_y": [],
        "bbox_x_length": [],
        "bbox_y_length": [],
        # "bbox_center_x_predicted": [],  # absent for brain data
        # "bbox_center_y_predicted": [],  # absent for brain data
        # "bbox_x_length_predicted": [],  # absent for brain data
        # "bbox_y_length_predicted": [],  # absent for brain data
        # "bbox_center_x_se": [],  # absent for brain data
        # "bbox_center_y_se": [],  # absent for brain data
        # "bbox_x_length_se": [],  # absent for brain data
        # "bbox_y_length_se": [],  # absent for brain data
        # "correct_classification": [],  # absent for brain data
    }

    print(f"\nProcessing category: {category}")
    # retrieve indices where the category is the desired one
    category_indices = np.where(responses_region["presentation"]["category_name"] == category)[0]
    # slice the dataset over those indices
    responses_category = lib.select(responses_region, "presentation", category_indices)

    # --- new block to collect raw repetitions ---
    responses_category_nonavg = lib.select(
        responses_nonavg_region,
        "presentation",
        np.where(responses_nonavg_region["presentation"]["category_name"] == category)[0]
    )

    # 1) get the *exact* image order from averaged data
    avg_image_ids = responses_category["presentation"]["image_id"]
    #    shape: (n_images,) in the exact order that your averaged values use

    # 2) pull the raw values and raw image_ids
    raw_vals = responses_category_nonavg["values"]  # shape (n_neurons, n_presentations)
    raw_image_ids = responses_category_nonavg["presentation"]["image_id"]  # (n_presentations,)

    # 3) build your list in that order
    values_with_reps = []
    for img_id in avg_image_ids:
        # find all repetition‐rows for this image
        idxs = np.where(raw_image_ids == img_id)[0]  # e.g. array([5,12,19]) if it repeated 3×
        group = raw_vals[:, idxs].T  # shape (n_repetitions, n_neurons)
        values_with_reps.append(group)
    # --------------------------------------------

    # STORE METADATA and SAVE IMAGES AND PKLs
    for i in range(len(responses_category["presentation"]["image_file_name"])):
        presentation["img_relative_path"].append(
            os.path.join(category, "images", os.path.splitext(responses_category['presentation']['image_file_name'][i])[0] + ".jpg"))
        presentation["category"].append(category)
        presentation["category_idx"].append(cat_to_idx[category])

        # Load image
        image = Image.open(os.path.join(images_folder_path, responses_category['presentation']['image_file_name'][i]))
        img_width, img_height = image.size
        # print(f"image size: {img_width}x{img_height}")

        # store jpg image for image dataset
        # Convert to RGB if needed (JPEG doesn't support transparency)
        image = image.convert("RGB")
        # Save it as JPEG
        jpg_img_path = os.path.join(dataset_location, output_image_dataset_name, category, "images",
                                    os.path.splitext(responses_category['presentation']['image_file_name'][i])[0] + ".jpg")
        image.save(jpg_img_path, format="JPEG")

        # Get bbox values
        x_min = responses_category['presentation']['axis_bb_left'][i]
        y_min = responses_category['presentation']['axis_bb_top'][i]
        x_max = responses_category['presentation']['axis_bb_right'][i]
        y_max = responses_category['presentation']['axis_bb_bottom'][i]

        # check if there is an image with coordinates outside bounds. Arbitrarily assign middle value of x_min and x_max
        if (
                x_min < 0 or x_max > img_width or x_min > x_max
                or y_min < 0 or y_max > img_height or y_min > y_max
        ):
            print("Detected clearly mislabeled image")
            print(f"Invalid coordinates: x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max}")
            x_min = int(2*img_width/6)
            x_max = int(4*img_width/6)
            y_min = int(2*img_height/6)
            y_max = int(4*img_height/6)
            print(f"New coordinates: x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max}")
            print("Set arbitrary new coordinates that do not interfere with dataset range")

        # store bbox value for images dataset
        bbox_dict = {"bbox_x_min": x_min, "bbox_x_max": x_max, "bbox_y_min": y_min, "bbox_y_max": y_max}
        bbox_path = os.path.join(dataset_location, output_image_dataset_name, category, "bboxes",
                                 os.path.splitext(responses_category['presentation']['image_file_name'][i])[0] + ".pkl")
        with open(bbox_path, "wb") as f:
            pickle.dump(bbox_dict, f)

        # infer bbox values for representations dataset and store them
        center_x = ((x_min + x_max) / 2.0) / img_width
        center_y = ((y_min + y_max) / 2.0) / img_height
        width = (x_max - x_min) / img_width
        height = (y_max - y_min) / img_height
        presentation["bbox_center_x"].append(center_x)
        presentation["bbox_center_y"].append(center_y)
        presentation["bbox_x_length"].append(width)
        presentation["bbox_y_length"].append(height)

    # STORE REPRESENTATIONS FILE
    result_dict = {
                "values": responses_category["values"].T,  # transposing to have the correct order # images, # neurons
                "values_with_repetitions": values_with_reps,  # list of length #images with arrays of shape [# repetitions, # neurons]
                "coordinate_positions": ["presentation", "neuroid"],
                "presentation": presentation
            }
    reps_path = os.path.join(dataset_location, output_representations_dataset_name, category, f"{category}.pkl")
    with open(reps_path, "wb") as f:
        pickle.dump(result_dict, f)

    # print(np.shape(result_dict["values"]))



