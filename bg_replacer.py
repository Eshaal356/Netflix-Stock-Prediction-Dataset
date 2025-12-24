from PIL import Image
import numpy as np

def process_image():
    input_path = "developer_white.png"
    output_path = "developer_black.png"
    
    try:
        img = Image.open(input_path).convert("RGBA")
        data = np.array(img)
        
        # Define white extraction threshold (tolerant to slight off-whites)
        threshold = 200
        
        r, g, b, a = data.T
        
        # Identify white areas
        white_areas = (r > threshold) & (g > threshold) & (b > threshold)
        
        # Replace white with Black (0, 0, 0, 255)
        data[..., :-1][white_areas.T] = (0, 0, 0)
        
        # Save result
        result = Image.fromarray(data)
        result.save(output_path)
        print("Success")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    process_image()
