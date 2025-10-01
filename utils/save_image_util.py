from PIL import Image
import numpy as np

# Function to save images in a grid layout
def save_images_grid(images, grid_size, output_path):
    """
    Saves a list of images in a grid layout to a single PNG file.
    
    Args:
        images (list of PIL.Image.Image): List of images to be saved.
        grid_size (tuple): Tuple specifying (rows, cols) for the grid.
        output_path (str): Path to save the output image file.
    """
    # Ensure the number of images matches the grid size
    assert len(images) == grid_size[0] * grid_size[1], "Number of images does not match grid size."

    # Get the dimensions of individual images
    img_width, img_height = images[0].size

    # Create a blank canvas for the grid
    grid_img = Image.new("RGB", (img_width * grid_size[1], img_height * grid_size[0]))

    # Paste images into the grid
    for i, img in enumerate(images):
        row = i // grid_size[1]
        col = i % grid_size[1]
        grid_img.paste(img, (col * img_width, row * img_height))

    # Save the grid image
    grid_img.save(output_path)
    
def save_images_grid_np(images, grid_size, output_path):
    """
    Saves a list of images (as NumPy arrays) in a grid layout to a single PNG file.
    
    Args:
        images (np.ndarray): A NumPy array of shape (num_images, height, width, channels).
        grid_size (tuple): Tuple specifying (rows, cols) for the grid.
        output_path (str): Path to save the output image file.
    """
    # Ensure the number of images matches the grid size
    num_images, img_height, img_width, num_channels = images.shape
    assert num_images == grid_size[0] * grid_size[1], "Number of images does not match grid size."

    # Create a blank canvas for the grid
    grid_img = np.zeros(
        (grid_size[0] * img_height, grid_size[1] * img_width, num_channels), dtype=np.uint8
    )

    # Fill the grid with images
    for idx, image in enumerate(images):
        row = idx // grid_size[1]
        col = idx % grid_size[1]
        start_y = row * img_height
        start_x = col * img_width
        grid_img[start_y:start_y + img_height, start_x:start_x + img_width, :] = image

    # Convert the grid to a PIL image and save
    grid_img = Image.fromarray(grid_img)
    grid_img.save(output_path)
