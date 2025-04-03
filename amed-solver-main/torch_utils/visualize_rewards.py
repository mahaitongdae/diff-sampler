import torch
import torchvision
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont


def draw_indices_on_images(images, indices, font_size = 12):
    """Overlay index numbers on each image in the batch."""
    images_with_text = []
    to_pil = T.ToPILImage()
    to_tensor = T.ToTensor()

    # Try to load a default font
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    for img_tensor, idx in zip(images, indices):
        img = to_pil(img_tensor)
        draw = ImageDraw.Draw(img)
        draw.text((2, 2), str(idx.item()), fill="white", font=font)
        images_with_text.append(to_tensor(img))

    return torch.stack(images_with_text)


# Example usage:
# # images: Tensor of shape (B, C, H, W)
# # indices: Tensor of shape (B,)
# def save_grid_with_indices(images, indices, filename = "grid.png"):
#     images_with_text = draw_indices_on_images(images, indices)
#     grid = torchvision.utils.make_grid(images_with_text, nrow=8, padding=2)
#     torchvision.utils.save_image(grid, filename)