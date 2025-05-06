from PIL import Image
import os

# Define paths
data_dir = 'lung_cancer_dataset'
normal_dir = os.path.join(data_dir, 'normal')
cancer_dir = os.path.join(data_dir, 'cancer')

# Ensure directories exist
if not os.path.exists(normal_dir) or not os.path.exists(cancer_dir):
    print(f"Error: One or both directories not found: {normal_dir}, {cancer_dir}")
    exit(1)

# Function to preprocess and save images
def preprocess_images(folder):
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder, filename)
            try:
                img = Image.open(img_path).convert('L')  # Convert to grayscale
                img = img.resize((64, 64))  # Resize to 64x64
                img.save(img_path, optimize=True)  # Overwrite with preprocessed image
                print(f"Processed: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")
        else:
            print(f"Skipped: {filename} (unsupported format)")

# Preprocess normal and cancer images
print("Processing normal images...")
preprocess_images(normal_dir)
print("Processing cancer images...")
preprocess_images(cancer_dir)
print("✅ All images preprocessed successfully!")