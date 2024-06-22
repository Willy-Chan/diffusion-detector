import os
import random
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel

# Define global variables for the number of masks
NUM_MASKS_LOWER = 2
NUM_MASKS_UPPER = 5

def apply_masks_and_inpaint(image, num_masks, size_of_mask):
    """ Apply random white patches and use inpainting to restore the image. """
    img_array = np.array(image)
    height, width, _ = img_array.shape
    mask = np.zeros((height, width), dtype=np.uint8)

    # Randomly place white patches which represent the mask
    for _ in range(num_masks):
        x = random.randint(0, width - size_of_mask)
        y = random.randint(0, height - size_of_mask)
        mask[y:y+size_of_mask, x:x+size_of_mask] = 255

    inpainted_image = cv2.inpaint(img_array, mask, inpaintRadius=3, flags=cv2.INPAINT_NS)

    # TODO: BETTER INPAINTING THAN JUST CV2???
    return Image.fromarray(cv2.cvtColor(inpainted_image, cv2.COLOR_BGR2RGB))

def load_clip_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    print(torch.cuda.is_available())
    return model, processor

def get_image_embeddings(image, model, processor):
    """ Prepare the image for the model and get embeddings. """
    inputs = processor(images=image, return_tensors="pt").to(model.device)
    outputs = model.get_image_features(**inputs)
    return outputs


### METRIC 1: DISTANCE!
def calculate_latent_change(original_image, inpainted_image, model, processor, significant_features, norm_degree=3):
    """ Calculate the cosine dissimilarity (1 - cosine similarity) using significant features between embeddings of original and inpainted images. """
    original_embedding = get_image_embeddings(original_image, model, processor)
    inpainted_embedding = get_image_embeddings(inpainted_image, model, processor)

    # Filter embeddings to include only significant features
    original_embedding = original_embedding[:, significant_features]
    inpainted_embedding = inpainted_embedding[:, significant_features]
    distance = torch.norm(original_embedding - inpainted_embedding, p=norm_degree).item()
    return distance  # Return the higher power norm distance



## METRIC 2: COSINE DISSIMILARITY!
import torch.nn.functional as F

# def calculate_latent_change(original_image, inpainted_image, model, processor, significant_features, norm_degree=3):
#     """ Calculate the cosine dissimilarity (1 - cosine similarity) using significant features between embeddings of original and inpainted images. """
#     original_embedding = get_image_embeddings(original_image, model, processor)
#     inpainted_embedding = get_image_embeddings(inpainted_image, model, processor)
    
#     # Filter embeddings to include only significant features
#     original_embedding = original_embedding[:, significant_features]
#     inpainted_embedding = inpainted_embedding[:, significant_features]
    
#     # Normalize the embeddings to unit vectors
#     original_embedding = F.normalize(original_embedding, p=2, dim=-1)
#     inpainted_embedding = F.normalize(inpainted_embedding, p=2, dim=-1)
    
#     # Calculate cosine similarity
#     cosine_similarity = (original_embedding * inpainted_embedding).sum(dim=-1)
    
#     # Calculate cosine dissimilarity
#     cosine_dissimilarity = 1 - cosine_similarity.item()  # Assuming batch size of 1
    
#     return cosine_dissimilarity

def find_optimal_threshold(results):
    """ Determine optimal threshold for classifying real vs. fake """
    all_values = results['Real'] + results['Fake']
    best_threshold, best_accuracy = 0, 0

    for threshold in np.linspace(min(all_values), max(all_values), num=100):
        predicted_real = [x < threshold for x in results['Real']]
        predicted_fake = [x >= threshold for x in results['Fake']]
        accuracy = (sum(predicted_real) + sum(predicted_fake)) / len(all_values)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    return best_threshold, best_accuracy


def analyze_folders(real_folder, fake_folder, model, processor):
    """ Analyze images from real and fake folders and plot histograms of latent space changes. """
    folders = {'Real': real_folder, 'Fake': fake_folder}
    results = {}

    for label, folder_path in folders.items():
        latent_changes = []
        count = 0
        for filename in sorted(os.listdir(folder_path)):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                if count >= 100:  # Process only the first 100 images
                    break
                image_path = os.path.join(folder_path, filename)
                original_image = Image.open(image_path).convert("RGB")

                ## NUMBER OF MASKS
                num_masks = (5, 10, 15)

                for nm in num_masks:
                    # basically inpaint, then use CLIP to make embeddings, then look at differences
                    inpainted_image = apply_masks_and_inpaint(original_image, nm, 90)
                    # Define a list of all unique significant features identified for both real and fake
                    significant_features = list(set([92, 258, 39, 45, 286, 321, 357, 106, 432, 389, 428, 137]))

                    # Convert to a PyTorch tensor for indexing
                    significant_features_tensor = torch.tensor(significant_features, dtype=torch.long)

                    # Now pass this tensor when calculating the latent change
                    distance = calculate_latent_change(original_image, inpainted_image, model, processor, significant_features_tensor)

                    # distance = calculate_latent_change(original_image, inpainted_image, model, processor, 5)  # LAST PARAM IS THE POWER OF THE DIFFERENCE!
                    latent_changes.append(distance)
                    count += 1

        results[label] = latent_changes


    threshold, accuracy = find_optimal_threshold(results)
    print(f"Optimal threshold: {threshold}, Accuracy: {accuracy}")

    # Plot histograms
    plt.figure(figsize=(10, 5))
    plt.hist(results['Real'], bins=30, alpha=0.5, label='Real', color='blue')
    plt.hist(results['Fake'], bins=30, alpha=0.5, label='Fake', color='red')
    plt.title('Histogram of Latent Space Changes Due to Inpainting')
    plt.xlabel('Latent Space Distance')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

def main():
    model, processor = load_clip_model()
    real_folder = "./aaa/test/real"
    fake_folder = "./aaa/test/fake"
    analyze_folders(real_folder, fake_folder, model, processor)

if __name__ == "__main__":
    main()
