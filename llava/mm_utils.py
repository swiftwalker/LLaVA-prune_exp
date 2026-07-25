from PIL import Image
from io import BytesIO
import base64
import torch
import math
import ast

from transformers import StoppingCriteria
from llava.constants import IMAGE_TOKEN_INDEX


def _config_value(config, name, default=None):
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def select_best_resolution(original_size, possible_resolutions):
    """
    Selects the best resolution from a list of possible resolutions based on the original size.

    Args:
        original_size (tuple): The original size of the image in the format (width, height).
        possible_resolutions (list): A list of possible resolutions in the format [(width1, height1), (width2, height2), ...].

    Returns:
        tuple: The best fit resolution in the format (width, height).
    """
    original_width, original_height = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float('inf')

    for width, height in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)

    return best_fit


def resize_and_pad_image(image, target_resolution):
    """
    Resize and pad an image to a target resolution while maintaining aspect ratio.

    Args:
        image (PIL.Image.Image): The input image.
        target_resolution (tuple): The target resolution (width, height) of the image.

    Returns:
        PIL.Image.Image: The resized and padded image.
    """
    original_width, original_height = image.size
    target_width, target_height = target_resolution

    scale_w = target_width / original_width
    scale_h = target_height / original_height

    if scale_w < scale_h:
        new_width = target_width
        new_height = min(math.ceil(original_height * scale_w), target_height)
    else:
        new_height = target_height
        new_width = min(math.ceil(original_width * scale_h), target_width)

    # Resize the image
    resized_image = image.resize((new_width, new_height))

    new_image = Image.new('RGB', (target_width, target_height), (0, 0, 0))
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    new_image.paste(resized_image, (paste_x, paste_y))

    return new_image


def divide_to_patches(image, patch_size):
    """
    Divides an image into patches of a specified size.

    Args:
        image (PIL.Image.Image): The input image.
        patch_size (int): The size of each patch.

    Returns:
        list: A list of PIL.Image.Image objects representing the patches.
    """
    patches = []
    width, height = image.size
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            box = (j, i, j + patch_size, i + patch_size)
            patch = image.crop(box)
            patches.append(patch)

    return patches


def get_anyres_image_grid_shape(image_size, grid_pinpoints, patch_size):
    """
    Calculate the shape of the image patch grid after the preprocessing for images of any resolution.

    Args:
        image_size (tuple): The size of the input image in the format (width, height).
        grid_pinpoints (str): A string representation of a list of possible resolutions.
        patch_size (int): The size of each image patch.

    Returns:
        tuple: The shape of the image patch grid in the format (width, height).
    """
    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    width, height = select_best_resolution(image_size, possible_resolutions)
    return width // patch_size, height // patch_size


def get_unpadded_feature_grid_shape(original_size, current_height, current_width):
    """Return the H/W left by ``unpad_image`` without materializing a tensor."""
    original_width, original_height = (int(original_size[0]), int(original_size[1]))
    current_height = int(current_height)
    current_width = int(current_width)
    if min(original_width, original_height, current_height, current_width) <= 0:
        raise ValueError(
            "Image and feature-grid dimensions must be positive, got "
            f"image={original_size}, grid=({current_height}, {current_width})"
        )

    if original_width / original_height > current_width / current_height:
        new_height = int(original_height * (current_width / original_width))
        padding = (current_height - new_height) // 2
        return current_height - 2 * padding, current_width

    new_width = int(original_width * (current_height / original_height))
    padding = (current_width - new_width) // 2
    return current_height, current_width - 2 * padding


def get_visual_token_layout(
    image_size,
    model_config,
    *,
    patches_per_side,
    vision_image_size,
    num_image_crops=None,
):
    """Describe the merged visual sequence produced by LLaVA multimodal prep.

    The returned counts follow ``prepare_inputs_labels_for_multimodal`` exactly:
    a base 24x24 view followed by the unpadded any-resolution tile grid, with
    one learned newline token appended to each local-grid row.
    """
    patches_per_side = int(patches_per_side)
    vision_image_size = int(vision_image_size)
    if patches_per_side <= 0 or vision_image_size <= 0:
        raise ValueError(
            "patches_per_side and vision_image_size must be positive, got "
            f"{patches_per_side}, {vision_image_size}"
        )

    base_tokens = patches_per_side * patches_per_side
    image_aspect_ratio = str(_config_value(model_config, "image_aspect_ratio", "square"))
    patch_merge_type = str(_config_value(model_config, "mm_patch_merge_type", "flat"))
    layout = {
        "layout_kind": "square",
        "visual_index_semantics": "merged_visual_sequence",
        "image_width": int(image_size[0]),
        "image_height": int(image_size[1]),
        "base_token_count": int(base_tokens),
        "local_patch_token_count": 0,
        "newline_token_count": 0,
        "local_grid_height": 0,
        "local_grid_width": 0,
        "total_token_count": int(base_tokens),
    }

    if image_aspect_ratio != "anyres":
        if num_image_crops not in (None, 1):
            raise ValueError(
                f"Non-anyres image should have one crop, got {num_image_crops}"
            )
        return layout

    grid_pinpoints = _config_value(model_config, "image_grid_pinpoints")
    if not grid_pinpoints:
        raise ValueError("anyres model config is missing image_grid_pinpoints")
    grid_width, grid_height = get_anyres_image_grid_shape(
        image_size,
        grid_pinpoints,
        vision_image_size,
    )
    expected_crops = 1 + int(grid_width) * int(grid_height)
    if num_image_crops is not None and int(num_image_crops) != expected_crops:
        raise ValueError(
            "Anyres crop count does not match the selected image grid: "
            f"got {num_image_crops}, expected {expected_crops} for grid "
            f"{grid_width}x{grid_height}"
        )

    local_height = int(grid_height) * patches_per_side
    local_width = int(grid_width) * patches_per_side
    newline_tokens = 0
    if patch_merge_type.startswith("spatial"):
        if "unpad" in patch_merge_type:
            local_height, local_width = get_unpadded_feature_grid_shape(
                image_size,
                local_height,
                local_width,
            )
            newline_tokens = local_height
        local_tokens = local_height * local_width
        total_tokens = base_tokens + local_tokens + newline_tokens
    elif patch_merge_type == "flat":
        local_tokens = (expected_crops - 1) * base_tokens
        local_height = int(grid_height) * patches_per_side
        local_width = int(grid_width) * patches_per_side
        total_tokens = expected_crops * base_tokens
    else:
        raise ValueError(f"Unexpected mm_patch_merge_type: {patch_merge_type}")

    layout.update(
        {
            "layout_kind": f"anyres_{patch_merge_type}",
            "crop_grid_width": int(grid_width),
            "crop_grid_height": int(grid_height),
            "num_image_crops": int(expected_crops),
            "local_patch_token_count": int(local_tokens),
            "newline_token_count": int(newline_tokens),
            "local_grid_height": int(local_height),
            "local_grid_width": int(local_width),
            "total_token_count": int(total_tokens),
        }
    )
    return layout


def process_anyres_image(image, processor, grid_pinpoints):
    """
    Process an image with variable resolutions.

    Args:
        image (PIL.Image.Image): The input image to be processed.
        processor: The image processor object.
        grid_pinpoints (str): A string representation of a list of possible resolutions.

    Returns:
        torch.Tensor: A tensor containing the processed image patches.
    """
    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    best_resolution = select_best_resolution(image.size, possible_resolutions)
    image_padded = resize_and_pad_image(image, best_resolution)

    patches = divide_to_patches(image_padded, processor.crop_size['height'])

    image_original_resize = image.resize((processor.size['shortest_edge'], processor.size['shortest_edge']))

    image_patches = [image_original_resize] + patches
    image_patches = [processor.preprocess(image_patch, return_tensors='pt')['pixel_values'][0]
                     for image_patch in image_patches]
    return torch.stack(image_patches, dim=0)


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def process_images(images, image_processor, model_cfg):
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    new_images = []
    if image_aspect_ratio == 'pad':
        for image in images:
            image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
            image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            new_images.append(image)
    elif image_aspect_ratio == "anyres":
        for image in images:
            image = process_anyres_image(image, image_processor, model_cfg.image_grid_pinpoints)
            new_images.append(image)
    else:
        return image_processor(images, return_tensors='pt')['pixel_values']
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images


def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]

class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        self.max_keyword_len = 0
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            if len(cur_keyword_ids) > self.max_keyword_len:
                self.max_keyword_len = len(cur_keyword_ids)
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]
    
    def call_for_batch(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        offset = min(output_ids.shape[1] - self.start_len, self.max_keyword_len)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            truncated_output_ids = output_ids[0, -keyword_id.shape[0]:]
            if torch.equal(truncated_output_ids, keyword_id):
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False
    
    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        outputs = []
        for i in range(output_ids.shape[0]):
            outputs.append(self.call_for_batch(output_ids[i].unsqueeze(0), scores))
        return all(outputs)
