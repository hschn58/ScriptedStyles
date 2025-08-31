from PIL import Image

def add_background(image_path, background_color, output_path):
    # Open the image
    image = Image.open(image_path).convert("RGBA")
    
    # Create a new image with the same size and the specified background color
    background = Image.new("RGBA", image.size, background_color)
    
    # Composite the image with the background
    combined = Image.alpha_composite(background, image)
    
    # Convert to RGB (to remove alpha channel) and save
    combined = combined.convert("RGB")
    combined.save(output_path)

# Usage
image_path = '/Users/henryschnieders/Desktop/proj/Designs/Banners/Etsy/2setsperpoint_trans2_highdpi copy 5.png'
background_color = (0, 0, 0, 255)  # Example: white background
output_path = image_path

add_background(image_path, background_color, output_path)
