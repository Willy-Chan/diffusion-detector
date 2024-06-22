import os
import random
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
import torch.nn.functional as F

def apply_masks_and_inpaint(image, num_masks, size_of_mask):
    """Apply random white patches and use inpainting to restore the image."""
    img_array = np.array(image)
    height, width, _ = img_array.shape
    mask = np.zeros((height, width), dtype=np.uint8)
    for _ in range(num_masks):
        x = random.randint(0, width - size_of_mask)
        y = random.randint(0, height - size_of_mask)
        mask[y:y+size_of_mask, x:x+size_of_mask] = 255
    inpainted_image = cv2.inpaint(img_array, mask, inpaintRadius=3, flags=cv2.INPAINT_NS)
    return Image.fromarray(cv2.cvtColor(inpainted_image, cv2.COLOR_BGR2RGB))

def load_clip_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    return model, processor

def get_image_embeddings(image, model, processor):
    """Prepare the image for the model and get embeddings."""
    inputs = processor(images=image, return_tensors="pt").to(model.device)
    outputs = model.get_image_features(**inputs)
    return outputs

def calculate_latent_change(original_image, inpainted_image, model, processor, significant_features):
    """Calculate the cosine dissimilarity using significant features between embeddings."""
    original_embedding = get_image_embeddings(original_image, model, processor)
    inpainted_embedding = get_image_embeddings(inpainted_image, model, processor)
    original_embedding = original_embedding[:, significant_features]
    inpainted_embedding = inpainted_embedding[:, significant_features]
    original_embedding = F.normalize(original_embedding, p=2, dim=-1)
    inpainted_embedding = F.normalize(inpainted_embedding, p=2, dim=-1)
    cosine_similarity = (original_embedding * inpainted_embedding).sum(dim=-1)
    cosine_dissimilarity = 1 - cosine_similarity.item()
    return cosine_dissimilarity

def find_optimal_threshold(real_changes, fake_changes):
    all_changes = np.concatenate([real_changes, fake_changes])
    thresholds = np.linspace(min(all_changes), max(all_changes), 100)
    best_threshold, best_accuracy = 0, 0
    for threshold in thresholds:
        real_correct = np.sum(real_changes > threshold)
        fake_correct = np.sum(fake_changes <= threshold)
        accuracy = (real_correct + fake_correct) / len(all_changes)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
    
    print(f"Best accuracy : {best_accuracy}")
    print(f"Best Threshold : {best_threshold}")
    return best_threshold, best_accuracy

def analyze_folders(real_folder, fake_folder, model, processor):
    results = {'Real': [], 'Fake': []}
    significant_features = torch.tensor([92, 258, 39, 45, 286, 321, 357, 106, 432, 389, 428, 137], dtype=torch.long)
    # significant_features = torch.tensor([92, 258, 39, 45, 286], dtype=torch.long)

    for label, folder_path in [('Real', real_folder), ('Fake', fake_folder)]:
        for filename in os.listdir(folder_path)[:100]:
            if filename.endswith((".jpg", ".png")):
                image_path = os.path.join(folder_path, filename)
                original_image = Image.open(image_path).convert("RGB")
                num_masks = (5, 10, 20)

                for nm in num_masks:
                    inpainted_image = apply_masks_and_inpaint(original_image, nm, 20)
                    distance = calculate_latent_change(original_image, inpainted_image, model, processor, significant_features)
                    results[label].append(distance)

    # Calculate and display the optimal threshold and accuracy
    real_distances = np.array(results['Real'])
    fake_distances = np.array(results['Fake'])
    optimal_threshold, optimal_accuracy = find_optimal_threshold(real_distances, fake_distances)

    # Plot histograms
    plt.figure(figsize=(12, 6))
    plt.hist(real_distances, bins=30, alpha=0.5, label='Real', color='blue')
    plt.hist(fake_distances, bins=30, alpha=0.5, label='Fake', color='red')
    plt.axvline(optimal_threshold, color='green', linestyle='dashed', linewidth=1)
    plt.title('Histogram of Cosine Dissimilarities')
    plt.xlabel('Cosine Dissimilarity')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

    print(f"Optimal threshold: {optimal_threshold}")
    print(f"Classification accuracy at optimal threshold: {optimal_accuracy*100:.2f}%")
    print(f"Variance - Real: {np.var(real_distances)}, Fake: {np.var(fake_distances)}")

def main():
    model, processor = load_clip_model()
    real_folder = "./aaa/test/real"
    fake_folder = "./aaa/test/fake"
    analyze_folders(real_folder, fake_folder, model, processor)

if __name__ == "__main__":
    main()
