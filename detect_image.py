import os
import random
import numpy as np
from PIL import Image
import cv2
import torch
from torchvision import transforms
from matplotlib import pyplot as plt
from diffusers import AutoencoderKL, StableDiffusionInpaintPipeline


NUM_MASKS_LOWER = 2
NUM_MASKS_UPPER = 3


# # EXPERIMENT WITH DIFFERENT MASKING/PERTURBATION TECHNIQUES
# # This CV2 inpainting shit performs horribly: the distributions look almost identical for me
# def apply_masks_and_inpaint(image, num_masks):
#     """ Apply random masks and inpaint the image """
#     img_array = np.array(image)
#     height, width, _ = img_array.shape
#     mask = np.zeros((height, width), dtype=np.uint8)

#     # Randomly place white patches
#     for _ in range(num_masks):
#         x = random.randint(0, width - 5)
#         y = random.randint(0, height - 5)
#         mask[y:y+5, x:x+5] = 255  # White patches for inpainting


#     # NEED SOMETHING BETTER THAN CV2 INPAINTING!
#     inpainted_img = cv2.inpaint(img_array, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)


#     return Image.fromarray(inpainted_img)


# JUST MASK OUT: Similar poor performance, histograms look the same!

# Input: image ()
# def apply_masks(image, num_masks):
#     """ Apply random white patches to the image without inpainting """
#     img_array = np.array(image)
#     height, width, _ = img_array.shape
#     mask = np.zeros((height, width), dtype=np.uint8)

#     # Randomly place white patches
#     for _ in range(num_masks):
#         x = random.randint(0, width - 5)
#         y = random.randint(0, height - 5)
#         img_array[y:y+5, x:x+5] = 255  # Apply white patches directly to the image

#     return Image.fromarray(img_array)


# INPAINTING WITH A PROMPT THAT IS DERIVED FROM A CNN DETECTION OF THE "MASKED OUT" PART, THEN NLP-TRANSLATED TO A COHERENT PROMPT









def apply_masks_and_inpaint(image, num_masks, model):
    """Apply random white patches and use Stable Diffusion to inpaint the image."""
    img_array = np.array(image)
    height, width, _ = img_array.shape
    mask = np.zeros((height, width), dtype=np.uint8)

    # Randomly place white patches which represent the mask
    for _ in range(num_masks):
        x = random.randint(0, width - 5)
        y = random.randint(0, height - 5)
        mask[y:y+5, x:x+5] = 1  # Using 1 to denote areas to inpaint

    # Convert image and mask to PIL Images for compatibility with Stable Diffusion pipeline
    image_pil = Image.fromarray(img_array)
    mask_pil = Image.fromarray(mask * 255).convert("L")  # Convert mask to 8-bit grayscale

    # Perform inpainting using the loaded model
    with torch.no_grad():
        output = model(prompt="", image=image_pil, mask_image=mask_pil)


    
    image_pil.save('original_image.jpg')
    mask_pil.save('masked_image.jpg')
    output["sample"][0].save('inpainted_image.jpg')
    
    return output["sample"][0]  # Assuming the model returns the inpainted image in a dict


def calculate_log_likelihood(vae, image):
    """ Calculate the VAE log likelihood for the given image """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image_tensor = transform(image).unsqueeze(0).to(vae.device)

    with torch.no_grad():
        # Encode the image to get the distribution
        output = vae.encode(image_tensor)
        latent = output.latent_dist.mean
        logvar = output.latent_dist.logvar  # Assuming the encoder also provides log variance

        # Decode the latent representation to reconstruct the image
        recon_image = vae.decode(latent)
        
        # Calculate reconstruction loss (Mean Squared Error)
        recon_loss = torch.nn.functional.mse_loss(recon_image.sample, image_tensor, reduction='sum')

        # Calculate KL Divergence between the latent distribution and a standard normal distribution
        kl_divergence = -0.5 * torch.sum(1 + logvar - latent.pow(2) - logvar.exp())

    # Calculate the negative log likelihood, which is the sum of the reconstruction loss and KL divergence
    log_px = -recon_loss.item() - kl_divergence.item()  # Convert to Python float for easier handling
    # log_px = -recon_loss.item() 
    return log_px # NOW ALSO INCLUDES KL DIVERGENCE TERM


def analyze_folders(real_folder, fake_folder, vae):
    """ Analyze images from real and fake folders and plot histograms """
    folders = {'Real': real_folder, 'Fake': fake_folder}
    results = {}

    for label, folder_path in folders.items():
        log_changes = []
        count = 0  # Initialize a counter to limit the number of processed images
        for filename in sorted(os.listdir(folder_path)):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                if count >= 50:  # Stop processing after 100 images
                    break
                image_path = os.path.join(folder_path, filename)
                original_image = Image.open(image_path).convert("RGB")

                ### KNOB 2: NUMBER OF MASKS
                # VARY THE LEVEL OF PERTURBATION / NUMBER OF MASKS
                # Apply a fixed level of perturbation
                for i in range(NUM_MASKS_LOWER,NUM_MASKS_UPPER + 1):
                    model = load_model()
                    inpainted_image = apply_masks_and_inpaint(original_image, i, model)  # Fixed to i masks


                    # inpainted_image = apply_masks(original_image, i)  # Fixed to i masks
                    original_ll = calculate_log_likelihood(vae, original_image)
                    perturbed_ll = calculate_log_likelihood(vae, inpainted_image)
                    log_changes.append(original_ll - perturbed_ll)
                count += 1
        
        results[label] = log_changes

    # Plot histograms
    plt.hist(results['Real'], bins=30, alpha=0.75, label='Real', color='blue')
    plt.hist(results['Fake'], bins=30, alpha=0.75, label='Fake', color='red')
    plt.title('Histogram of Log-Likelihood Changes for Real vs Fake Images')
    plt.xlabel('Change in Log-Likelihood')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

def main():
    vae_model = "stabilityai/sd-vae-ft-mse"

    ### KNOB 3: VAE MODEL
    vae = AutoencoderKL.from_pretrained(vae_model).to('cuda' if torch.cuda.is_available() else 'cpu')
    # real_folder = "./archive/test/REAL"
    # fake_folder = "./archive/test/FAKE"
    # analyze_folders(real_folder, fake_folder, vae)



if __name__ == "__main__":
    main()
