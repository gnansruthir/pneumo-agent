import os
from PIL import Image, ImageDraw, ImageFilter

def create_mock_xray(filename, abnormal_type=None):
    # Create a 512x512 grayscale canvas representing a chest x-ray background
    img = Image.new("L", (512, 512), color=20)
    draw = ImageDraw.Draw(img)
    
    # Draw chest cage outline (ribs effect / spine)
    # Spine (vertical center line)
    draw.rectangle([250, 60, 262, 450], fill=65)
    
    # Ribs (horizontal lines curving downwards)
    for y in range(100, 420, 35):
        # Left ribs
        draw.arc([100, y - 20, 250, y + 40], start=180, end=270, fill=45, width=6)
        # Right ribs
        draw.arc([262, y - 20, 412, y + 40], start=270, end=360, fill=45, width=6)
        
    # Draw dark lung cavities (left and right)
    # Left lung cavity
    draw.ellipse([120, 100, 240, 400], fill=10)
    # Right lung cavity
    draw.ellipse([272, 100, 392, 400], fill=10)
    
    # Draw collar bones
    draw.line([100, 90, 248, 120], fill=80, width=12)
    draw.line([412, 90, 264, 120], fill=80, width=12)
    
    # Add pathology abnormalities
    if abnormal_type == "tuberculosis":
        # Upper lobe infiltrates/cavity in the right lung (left side of image, index coords)
        draw.ellipse([140, 140, 180, 180], fill=90)
        draw.ellipse([145, 145, 175, 175], fill=10) # Cavity core
        draw.ellipse([185, 170, 205, 190], fill=75) # Nodule
    elif abnormal_type == "pneumothorax":
        # Pleural line showing collapsed left lung (right side of image, index coords)
        # We fill the outer edge with pure black to show lack of lung markings
        draw.chord([272, 100, 392, 400], start=270, end=90, fill=5)
        # Draw thin white visceral pleural line
        draw.arc([300, 110, 370, 390], start=270, end=90, fill=110, width=2)
        
    # Apply heavy blur to make it look like a smooth continuous X-ray image
    blurred_img = img.filter(ImageFilter.GaussianBlur(radius=8))
    
    # Add a bit of noise to simulate scanner grain
    import numpy as np
    img_data = np.array(blurred_img)
    noise = np.random.normal(0, 5, img_data.shape)
    noisy_img_data = np.clip(img_data + noise, 0, 255).astype(np.uint8)
    
    final_img = Image.fromarray(noisy_img_data)
    final_img.save(filename)
    print(f"Generated sample: {filename}")

if __name__ == "__main__":
    samples_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(samples_dir, exist_ok=True)
    
    create_mock_xray(os.path.join(samples_dir, "sample_normal.png"))
    create_mock_xray(os.path.join(samples_dir, "sample_tuberculosis.png"), "tuberculosis")
    create_mock_xray(os.path.join(samples_dir, "sample_pneumothorax.png"), "pneumothorax")
