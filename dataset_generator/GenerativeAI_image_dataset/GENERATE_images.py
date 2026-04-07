import sys
import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from diffusers import LCMScheduler, AutoPipelineForText2Image, AutoPipelineForInpainting
import random
import time
from PIL import ImageFilter
import matplotlib
import matplotlib.pyplot as plt
import pickle
import os
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import argparse
import glob

# PARAMETERS
fontsize = 40
batch_size_generation = 5
batch_size_inpainting = 10
batches_per_category = 1800  # 9000/5=1800 # total number of generations will be generations_per_category * batch sizes
new_bbox_max_length_min = 200  # should be < then inpainting size
new_bbox_max_length_max = 500  # should be < then inpainting size
categories_dictionary_location = "path/to/Objects365_categories.pkl"  # Contains generation_name and detection_names for each category.
# CerberusDet location:
sys.path.append("path/to/CerberusDet")  # Directory that exposes the cerberusdet Python package.
cerberus_weights_location = "path/to/voc_obj365_full_bs_best.pt"  # CerberusDet checkpoint trained on Objects365.
results_location = "path/to/output_dataset"  # Output root; one subdirectory per category will be created here.


# DEFINE PROMPT
def generate_prompt(category):
    prompt = (f"A full-body, photorealistic shot of a single {category}. "
              f"The {category} is the clear main subject, fully in the scene, and unobstructed. "
              f"The background is detailed, contextually appropriate, but does not distract.")
    negative_prompt = (
        "clutter, multiple objects, occlusion, low quality, blurry, distorted, unrealistic, extra limbs, cropped, artifacts")

    if category != "person":
        negative_prompt += ", persons, people, humans, mannequin"

    return prompt, negative_prompt


# CONSTANTS
img_generation_size = (1024, 1024)
image_detection_size = (640, 640)
img_inpainting_size = (512, 512)
color_scheme = ["red", "green", "blue", "yellow", "magenta", "cyan"]

# RETRIEVE JOB_ID
# Set up argument parsing
parser = argparse.ArgumentParser(description="Run SDXL generation with SLURM job ID.")
parser.add_argument("--job_id", type=str, default="00000", help="SLURM job ID")
# Parse arguments
args = parser.parse_args()
job_id = args.job_id

from cerberusdet.models.cerberus import CerberusDet
from cerberusdet.models.experimental import attempt_load
from cerberusdet.utils.general import non_max_suppression, scale_boxes
from cerberusdet.utils.torch_utils import select_device

# LOAD THE CATEGORIES DICTIONARIES
with open(categories_dictionary_location, 'rb') as f:
    categories_dictionary = pickle.load(f)

# LOAD MODELS
device = select_device("cuda" if torch.cuda.is_available() else "cpu")

# Load CerberusDet
print("Loading CerberusDet...", flush=True)
model: CerberusDet = attempt_load(
    cerberus_weights_location,
    map_location=device)
model.eval()
objects365_classes = model.names["objects365_full"]

# LOAD SDXL-Lightening
print("Loading SDXL-lightening...", flush=True)
base = "stabilityai/stable-diffusion-xl-base-1.0"
repo = "ByteDance/SDXL-Lightning"
ckpt = "sdxl_lightning_4step_unet.safetensors"  # Use the correct ckpt for your step setting!

# Load model.
unet = UNet2DConditionModel.from_config(base, subfolder="unet").to(device, torch.float16)
unet.load_state_dict(load_file(hf_hub_download(repo, ckpt)))
pipe = StableDiffusionXLPipeline.from_pretrained(base, unet=unet, torch_dtype=torch.float16, variant="fp16").to(device)

# Ensure sampler uses "trailing" timesteps.
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

# Disable the NSFW checker
pipe.safety_checker = None

# pipe.enable_xformers_memory_efficient_attention()

# Load Inpainting Pipeline
print("Loading SD1.5 inpainting...", flush=True)
pipe_inpaint = AutoPipelineForInpainting.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-inpainting",
    torch_dtype=torch.float16,
    variant="fp16",
).to("cuda")

# set scheduler
pipe_inpaint.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

# load LCM-LoRA
pipe_inpaint.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
pipe_inpaint.fuse_lora()

# Disable the NSFW checker
pipe_inpaint.safety_checker = None

# DEBUG: replace custom category dictionary
# categories_dictionary = {"Monkey": {"generation_name": ["monkey"], "detection_names": ["Monkey", "Bear", "Cat", "Person"]}}
# END DEBUG

number_categories = len(categories_dictionary)
# shuffle order of categories, so each job likely processes a different category
category_subdictionaries = list(categories_dictionary.values())
random.shuffle(category_subdictionaries)
for n, category_dict in enumerate(category_subdictionaries):

    # define category folder
    category_folder = os.path.join(results_location, category_dict['generation_name'][0].replace(" ", "_"))
    # create the category folder
    os.makedirs(category_folder, exist_ok=True)
    # create the images and bboxes subfolders
    images_folder = category_folder + "/images"
    bboxes_folder = category_folder + "/bboxes"
    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(bboxes_folder, exist_ok=True)

    # check whether there are already the desired number of images (these may have come from multiple, killed jobs)
    n_existing_images = sum(1 for entry in os.scandir(images_folder) if entry.is_file() and entry.name.endswith('.jpg'))
    if n_existing_images >= batch_size_generation*batches_per_category:
        print(f"Found {n_existing_images} >= {batch_size_generation*batches_per_category} "
              f"images for {category_dict['generation_name'][0]}. "
              f"Skipping this category", flush=True)
        continue
    else:
        print(f"Found {n_existing_images} < {batch_size_generation*batches_per_category} "
              f"images for {category_dict['generation_name'][0]}. "
              f"Proceeding generating more images.", flush=True)

    print(f"Processing category: {category_dict['generation_name'][0]}, {n + 1}/{number_categories}", flush=True)
    prompt, negative_prompt = generate_prompt(category_dict['generation_name'][0])
    inpaint_prompt = (
        f"{prompt} "
        "The background extends naturally from the existing scene. "
        "Seamless transition between regions, no visible boundaries. "
        "Consistent lighting and focus throughout the entire image."
    )

    success_stage_1_images_collector = []

    stage1_success_bounding_boxes_collector = []

    success_image_id_number = 1
    n_success_stage_1 = 0
    n_success_stage_2 = 0

    # Start timing
    start_time = time.time()

    while n_success_stage_2 < (batch_size_generation*batches_per_category - n_existing_images):

        # DEBUG: Start timing
        # start_time_batch = time.time()

        # Generate images
        images = pipe(
            prompt=[prompt] * batch_size_generation,  # Pass the same prompt `num_images` times
            negative_prompt=[negative_prompt] * batch_size_generation,
            num_inference_steps=4,
            guidance_scale=0).images

        # <editor-fold desc="Detect with Cerberus">
        # Convert images to tensors
        image_tensors = torch.stack([
            torch.from_numpy(cv2.resize(np.array(image), image_detection_size)).permute(2, 0, 1).float().div(255)
            for image in images
        ]).to(device)  # Shape: (batch_size, 3, 640, 640)

        # Run model inference
        with torch.no_grad():
            preds = model(image_tensors)["objects365_full"]

        # Apply Non-Maximum Suppression (NMS)
        batch_detections = non_max_suppression(preds, conf_thres=0.25, iou_thres=0.45)
        # </editor-fold>

        # <editor-fold desc="Process detections">
        # Rescale bounding boxes to match original image size
        for detections in batch_detections:
            if detections is not None:
                detections[:, :4] = scale_boxes(image_detection_size, detections[:, :4], img_generation_size)

        # check which images passed the detection test (a single instance of the category to be detected)
        for i, detections in enumerate(batch_detections):
            detected_categories = [objects365_classes[int(d[5])] for d in detections] if detections is not None else []
            main_category_detected = None
            test_passed = False

            # Check if at least one relevant category is detected
            for cat in category_dict['detection_names']:
                if cat in detected_categories:
                    main_category_detected = cat
                    break

            if main_category_detected:
                # Count occurrences of main_category_detected
                count_main_category = detected_categories.count(main_category_detected)

                # The test passes if exactly one instance is detected
                test_passed = count_main_category == 1

            if test_passed:
                success_stage_1_images_collector.append(images[i])
                n_success_stage_1 += 1

                # Find the bounding box for main_category_detected
                for d in detections:
                    if objects365_classes[int(d[5])] == main_category_detected:
                        stage1_success_bounding_boxes_collector.append(d[:4].tolist())  # Append bounding box coordinates
                        break  # Ensure only the first detected instance is stored
        # </editor-fold>

        # DEBUG: End timing
        # end_time_batch = time.time()
        # print(
        #     f"batch of size {batch_size_generation} generated and detected in: {end_time_batch - start_time_batch:.3f} seconds",
        #     flush=True)

        # proceed with inpainting
        # [if I have enough success images to fill a batch, or if I am at the last loop]
        if len(success_stage_1_images_collector) >= batch_size_inpainting:

            # DEBUG: Start timing
            # start_time_inpaint = time.time()

            # gather images to be inpainted
            success_stage_1_images = success_stage_1_images_collector[:batch_size_inpainting]
            stage1_success_bounding_boxes = stage1_success_bounding_boxes_collector[:batch_size_inpainting]
            # Remove the extracted elements from the collector lists
            success_stage_1_images_collector = success_stage_1_images_collector[batch_size_inpainting:]
            stage1_success_bounding_boxes_collector = stage1_success_bounding_boxes_collector[batch_size_inpainting:]

            # <editor-fold desc="RESAMPLE BBOX, GENERATE MASK AND PADDED IMAGES">
            masks = []
            padded_images = []
            for image, bbox in zip(success_stage_1_images, stage1_success_bounding_boxes):
                # sample new max length of the bbox
                new_bbox_max_length = random.randint(new_bbox_max_length_min, new_bbox_max_length_max)
                # DEBUG (ok for generation success):
                # new_bbox_max_length = random.randint(200, 200)
                # END DEBUG

                # rescale bbox to new size
                orig_bbox_w = bbox[2] - bbox[0]
                orig_bbox_h = bbox[3] - bbox[1]
                scale_factor = new_bbox_max_length / max(orig_bbox_w, orig_bbox_h)
                new_bbox_w, new_bbox_h = int(orig_bbox_w * scale_factor), int(orig_bbox_h * scale_factor)

                # rescale the original image
                orig_w, orig_h = image.size
                new_w, new_h = int(orig_w * scale_factor), int(orig_h * scale_factor)
                rescaled_image = image.resize((new_w, new_h), Image.LANCZOS)
                rescaled_bbox_x_min = int(bbox[0] * scale_factor)
                rescaled_bbox_y_min = int(bbox[1] * scale_factor)

                # sample new bbox center
                new_bbox_x_center = random.randint(new_bbox_w // 2, img_inpainting_size[0] - new_bbox_w // 2)
                new_bbox_y_center = random.randint(new_bbox_h // 2, img_inpainting_size[0] - new_bbox_h // 2)

                # convert to new bbox corners coordinates
                new_bbox_x_min, new_bbox_y_min = new_bbox_x_center - new_bbox_w // 2, new_bbox_y_center - new_bbox_h // 2
                new_bbox_x_max, new_bbox_y_max = new_bbox_x_min + new_bbox_w, new_bbox_y_min + new_bbox_h

                # past original generated image in the new position on blank canvas
                canvas = Image.new("RGB", img_inpainting_size, (255, 255, 255))
                paste_x = new_bbox_x_min - rescaled_bbox_x_min
                paste_y = new_bbox_y_min - rescaled_bbox_y_min

                # -> determine coordinates of white regions
                white_left = max(0, paste_x)
                white_top = max(0, paste_y)
                white_right = min(img_inpainting_size[0], new_w + paste_x)
                white_bottom = min(img_inpainting_size[0], new_h + paste_y)

                # -> paste
                canvas.paste(rescaled_image, (int(paste_x), int(paste_y)))

                # generate mask
                mask = np.ones(img_inpainting_size, dtype=np.uint8) * 255
                mask[white_top + 1:white_bottom - 1, white_left + 1:white_right - 1] = 0
                mask = Image.fromarray(mask)
                mask = mask.filter(ImageFilter.GaussianBlur(radius=20))  # Soften mask edges
                mask = mask.convert("L")  # Ensure grayscale mode
                masks.append(mask)

                # <editor-fold desc="APPLY REFLECTION PADDING:">
                blur_width = 0
                blur_height = 0
                # <editor-fold desc="Width right">
                if white_right < img_inpainting_size[0]:
                    # Extract the valid column for reflection
                    if new_bbox_x_max < white_right:
                        column = np.array(canvas)[:, new_bbox_x_max:white_right, :]
                    else:
                        column = np.array(canvas)[:, white_right - 1:white_right, :]  # Single column

                    if column.shape[1] < 10:
                        blur_radius = blur_width
                    else:
                        blur_radius = 0

                    flipped_column = np.flip(column, axis=1)  # Flip horizontally

                    # Compute number of times to tile the pattern
                    repeats = (img_inpainting_size[0] - white_right) // column.shape[1] + 1

                    # Alternate tiling of column and flipped_column
                    alternating_pattern = np.concatenate(
                        [flipped_column if i % 2 == 0 else column for i in range(repeats)], axis=1
                    )

                    # # Trim to fit within the blank region
                    # filled_region = alternating_pattern[:, :img_inpainting_size[0] - white_right, :]

                    # Convert back to image and paste on canvas
                    filled_region_image = Image.fromarray(alternating_pattern.astype(np.uint8))

                    filled_region_image = filled_region_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))

                    canvas.paste(filled_region_image, (white_right, 0))
                # </editor-fold>

                # <editor-fold desc="Width left">
                if white_left > 0:
                    # Extract the valid column for reflection
                    if new_bbox_x_min > white_left:
                        column = np.array(canvas)[:, white_left:new_bbox_x_min, :]
                    else:
                        column = np.array(canvas)[:, white_left:white_left + 1, :]  # Single column

                    if column.shape[1] < 10:
                        blur_radius = 3
                    else:
                        blur_radius = 0

                    flipped_column = np.flip(column, axis=1)  # Flip horizontally

                    # Compute number of times to tile the pattern
                    repeats = (white_left - 0) // column.shape[1] + 1

                    # Alternate tiling of column and flipped_column
                    alternating_pattern = np.concatenate(
                        [column if i % 2 == 0 else flipped_column for i in range(repeats)], axis=1
                    )

                    alternating_pattern = np.flip(alternating_pattern, axis=1)

                    # Convert back to image and paste on canvas
                    filled_region_image = Image.fromarray(alternating_pattern.astype(np.uint8))

                    filled_region_image = filled_region_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))

                    canvas.paste(filled_region_image, (white_left - np.shape(alternating_pattern)[1], 0))
                # </editor-fold>

                # <editor-fold desc="Height top">
                if white_top > 0:
                    # Extract the valid column for reflection
                    # if new_bbox_y_min > white_top:
                    #     column = np.array(canvas)[white_top:new_bbox_y_min, :, :]
                    # else:
                    #     column = np.array(canvas)[white_top:white_top+1, :, :]  # Single column
                    #
                    column = np.array(canvas)[white_top:white_top + 1, :, :]  # Single column

                    flipped_column = np.flip(column, axis=0)

                    # Compute number of times to tile the pattern
                    repeats = (white_top - 0) // column.shape[0] + 1

                    # Alternate tiling of column and flipped_column
                    alternating_pattern = np.concatenate(
                        [column if i % 2 == 0 else flipped_column for i in range(repeats)], axis=0
                    )

                    alternating_pattern = np.flip(alternating_pattern, axis=0)

                    # Convert back to image and paste on canvas
                    filled_region_image = Image.fromarray(alternating_pattern.astype(np.uint8))

                    filled_region_image = filled_region_image.filter(ImageFilter.GaussianBlur(radius=blur_height))

                    canvas.paste(filled_region_image, (0, white_top - np.shape(alternating_pattern)[0]))
                # </editor-fold>

                # <editor-fold desc="Height bottom">
                if white_bottom < img_inpainting_size[0]:
                    # Extract the valid column for reflection
                    if new_bbox_y_max < white_bottom:
                        column = np.array(canvas)[new_bbox_y_max:white_bottom, :, :]
                    else:
                        column = np.array(canvas)[white_bottom - 1:white_bottom, :, :]  # Single column

                    # column = np.array(canvas)[white_bottom-1:white_bottom, :, :]  # Single column

                    flipped_column = np.flip(column, axis=0)

                    # Compute number of times to tile the pattern
                    repeats = (img_inpainting_size[0] - white_bottom) // column.shape[0] + 1

                    # Alternate tiling of column and flipped_column
                    alternating_pattern = np.concatenate(
                        [column if i % 2 == 0 else flipped_column for i in range(repeats)], axis=0
                    )

                    alternating_pattern = np.flip(alternating_pattern, axis=0)

                    # Convert back to image and paste on canvas
                    filled_region_image = Image.fromarray(alternating_pattern.astype(np.uint8))

                    filled_region_image = filled_region_image.filter(ImageFilter.GaussianBlur(radius=blur_height))

                    canvas.paste(filled_region_image, (0, white_bottom))
                # </editor-fold>

                padded_images.append(canvas)
                # </editor-fold>
            # </editor-fold>

            success_stage_2_images = []
            success_stage_2_bboxes = []

            # GENERATE OUTPAINTED IMAGES
            images = pipe_inpaint(
                prompt=[inpaint_prompt] * len(masks),
                negative_prompt=[negative_prompt] * len(masks),
                image=padded_images,
                mask_image=masks,
                num_inference_steps=4,
                strength=0.95,
                guidance_scale=0,
            ).images

            # <editor-fold desc="Detect with Cerberus">
            # Convert images to tensors
            image_tensors = torch.stack([
                torch.from_numpy(cv2.resize(np.array(image), image_detection_size)).permute(2, 0, 1).float().div(
                    255)
                for image in images
            ]).to(device)  # Shape: (batch_size, 3, 640, 640)

            # Run model inference
            with torch.no_grad():
                preds = model(image_tensors)["objects365_full"]

            # Apply Non-Maximum Suppression (NMS)
            batch_detections = non_max_suppression(preds, conf_thres=0.25, iou_thres=0.45)
            # </editor-fold>

            # <editor-fold desc="Process detections">
            # Rescale bounding boxes to match original image size
            for detections in batch_detections:
                if detections is not None:
                    detections[:, :4] = scale_boxes(image_detection_size, detections[:, :4], img_inpainting_size)

            # Check which images passed the detection test (a single instance of the category to be detected)
            for i, detections in enumerate(batch_detections):
                detected_categories = [objects365_classes[int(d[5])] for d in
                                       detections] if detections is not None else []
                main_category_detected = None
                test_passed = False

                # Check if at least one relevant category is detected
                for cat in category_dict['detection_names']:
                    if cat in detected_categories:
                        main_category_detected = cat
                        break

                if main_category_detected:
                    # Count occurrences of main_category_detected
                    count_main_category = detected_categories.count(main_category_detected)

                    # The test passes if exactly one instance is detected
                    test_passed = count_main_category == 1

                if test_passed:
                    success_stage_2_images.append(images[i])
                    n_success_stage_2 += 1
                    # Find the bounding box for main_category_detected
                    for d in detections:
                        if objects365_classes[int(d[5])] == main_category_detected:
                            success_stage_2_bboxes.append(
                                d[:4].tolist())  # Append bounding box coordinates
                            break  # Ensure only the first detected instance is stored

            # </editor-fold>

            # SAVE IMAGES AND BBOXES
            for image, bbox in zip(success_stage_2_images, success_stage_2_bboxes):
                # define image unique name
                name = f"{category_dict['generation_name'][0].replace(' ', '_')}_{job_id}_{success_image_id_number}"

                # same image
                image.convert("RGB").save(os.path.join(images_folder, name + ".jpg"), "JPEG", quality=95)

                # save bbox
                x_min = bbox[0]
                x_max = bbox[2]
                y_min = bbox[1]
                y_max = bbox[3]
                bbox_dict = {"bbox_x_min": x_min, "bbox_x_max": x_max, "bbox_y_min": y_min, "bbox_y_max": y_max}
                with open(os.path.join(bboxes_folder, name + ".pkl"), "wb") as f:
                    pickle.dump(bbox_dict, f)

                # update image number id
                success_image_id_number += 1

            # DEBUG: End timing
            # end_time_inpaint = time.time()
            # print(
            #     f"batch of size {batch_size_inpainting} inpainted, detected, stored in: {end_time_inpaint - start_time_inpaint:.3f} seconds",
            #     flush=True)

    # End timing
    end_time = time.time()

    # Print the elapsed time in seconds
    print(f"Images generation in: {(end_time - start_time)/3600:.4f} hrs", flush=True)

    # Print generation success statistics
    total_generated_images = batch_size_generation*batches_per_category

    print(f"Success rate: {100*n_success_stage_2 / total_generated_images}%", flush=True)
    print(f"Stage 1: {100 * n_success_stage_1 / total_generated_images}%", flush=True)
    print(f"Stage 2: {100 * n_success_stage_2 / n_success_stage_1}%", flush=True)

    # save "DONE" dictionary with generation success statistics
    success_dictionary = {"total_success_rate": 100*n_success_stage_2 / total_generated_images,
                          "stage_1_success_rate": 100 * n_success_stage_1 / total_generated_images,
                          "stage_2_success_rate": 100 * n_success_stage_2 / n_success_stage_1,
                          "stage_1_generated_images": total_generated_images}

    dict_name = f"{job_id}_done.pkl"
    with open(os.path.join(category_folder, dict_name), "wb") as f:
        pickle.dump(success_dictionary, f)


