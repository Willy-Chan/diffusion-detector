import torch
from PIL import Image
from torchvision.transforms import ToTensor, Resize, ToPILImage, Compose, Normalize
from diffusers import UNet2DModel, DDPMScheduler
import os
import matplotlib.pyplot as plt
from pytorch_diffusion import Diffusion

# Function to load and preprocess an image
def load_and_preprocess_image(filepath, image_size=256):
    image = Image.open(filepath).convert("RGB")
    transform = Compose([
        Resize((image_size, image_size)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Typical normalization for pretrained models
    ])
    image = transform(image)
    return image.unsqueeze(0)  # Add batch dimension

# Function to show images
def show_images(images):
    to_pil = ToPILImage()
    for img in images:
        # Undo normalization for displaying
        img = img * torch.tensor([0.229, 0.224, 0.225])[:, None, None] + torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        to_pil(img).show()

def get_noised_images(input_tensor, scheduler, noise, timesteps):
    noised_images = scheduler.add_noise(input_tensor, noise, timesteps)

    # need to normalize the images
    normalized_images = (noised_images * 0.5 + 0.5).clip(0, 1)
    return normalized_images
    

"""
Ignore this function: only used for custom trained unet2D model
"""
def get_accuracy_scores(input_filepath, scheduler, timesteps, noise, model):
    for i, file in enumerate(os.listdir(input_filepath)):
        image_tensor = load_and_preprocess_image(input_filepath + file)
        noised_images_tensor = get_noised_images(image_tensor, scheduler, noise, timesteps)

        with torch.no_grad():
            for i, (noisy_image, timestep) in enumerate(zip(noised_images_tensor, timesteps)):
                
                noisy_image = noisy_image.unsqueeze(0)   # Add a batch dimension to the noisy image
                noise_pred = model(noisy_image, timestep=timestep)[0]
                print(f"Image {i}, Timestep {timestep.item()}: Predicted noise level: {noise_pred.mean().item()}")



def get_noise_level_values(timesteps, input_image_tensor):
    noise_levels = torch.zeros_like(timesteps).float()     # tensor that stores all of the "noise level" measures

    with torch.no_grad():
        for i, t in enumerate(timesteps):
            t_tensor = torch.tensor([t], device=diffusion.device).repeat(input_image_tensor.size(0))
            model_output = diffusion.model(input_image_tensor, t_tensor)
            noise_levels[i] = model_output.abs().mean()

    return noise_levels
        




if __name__ == '__main__':
    # image_url = "example.png"
    # image_tensor = load_and_preprocess_image(image_url)

    """
    Part 0: Load
    """
    

    """
    Part 1: Noise the images and get a tensor of a bunch of noised images

    Loop over each of the noised images, look at the UNet score, and evaluate and classify
    """

    if None:
        ### Code for using a custom trained UNet2DModel
        net = UNet2DModel(
        in_channels=3,  # RGB images
        out_channels=3,  # Output also RGB
        sample_size=512,  # Same as image size after resize
        block_out_channels=(32, 64, 128, 256),
        norm_num_groups=8
        )
        net.eval()
        noised_image_tensor = get_noised_images(input_tensor=image_tensor,
                                                scheduler=DDPMScheduler(beta_start=0.0001, beta_end=0.02),
                                                noise=torch.randn_like(image_tensor),
                                                timesteps=torch.linspace(0, 999, 8).long())

        #   show_images(normalized_images)  # Display the noised images
        # Simulate model inference to predict the noise level
        with torch.no_grad():
            for i, (noisy_image, timestep) in enumerate(zip(noised_images, timesteps)):
                # Dummy class labels, for demonstration only
                class_labels = torch.tensor([0])  # Assuming class '0' for all
                # Predict noise level; .sample() to simulate fetching model output
                noise_pred = net(noisy_image.unsqueeze(0), timestep=timestep, class_labels=class_labels)[0]
                print(f"Image {i}, Timestep {timestep.item()}: Predicted noise level: {noise_pred.mean().item()}")

        ex_image_tensor = load_and_preprocess_image(FILE_TO_REAL_IMAGES + os.listdir(FILE_TO_REAL_IMAGES)[0], image_size=diffusion.model.resolution)
        noise = torch.randn_like(ex_image_tensor)
        ex_image_tensor = ex_image_tensor.to(diffusion.device)
        get_accuracy_scores(FILE_TO_REAL_IMAGES, 
                            scheduler=DDPMScheduler(beta_start=0.0001, beta_end=0.02),
                            timesteps=torch.linspace(0, 999, 8).long(),
                            noise=noise,
                            model=net)



    
    FILE_TO_REAL_IMAGES = './test/REAL/'
    FILE_TO_FAKE_IMAGES = './test/FAKE/'
    diffusion = Diffusion.from_pretrained("cifar10")

    
    # Since you want the output directly from the model and not the complete denoising step,
    # You might need to manually handle the timestep and noise calculations
    # For simplicity, let's assume you want to evaluate it at a specific timestep
    # timestep = torch.tensor([0], device=diffusion.device)  # Example: very first timestep
    # noise = torch.randn_like(ex_image_tensor).to(diffusion.device)
    timesteps = torch.linspace(0, 999, 1000).long()  # Assuming 1000 timesteps


    NUM_IMAGES_TO_ANALYZE = 1

    for i, file in enumerate(os.listdir(FILE_TO_REAL_IMAGES)[:NUM_IMAGES_TO_ANALYZE]):  # Assuming you want only the first file for simplicity
        filepath = os.path.join(FILE_TO_REAL_IMAGES, file)
        input_image_tensor = load_and_preprocess_image(filepath, image_size=diffusion.model.resolution).to(diffusion.device)
        image_noise_levels = get_noise_level_values(timesteps, input_image_tensor)
        plt.plot(timesteps.cpu().numpy(), image_noise_levels.cpu().numpy(), color='blue', label='Real' if i == 0 else "")

    for i, file in enumerate(os.listdir(FILE_TO_FAKE_IMAGES)[:NUM_IMAGES_TO_ANALYZE]):  # Similarly, only the first file for simplicity
        filepath = os.path.join(FILE_TO_FAKE_IMAGES, file)
        input_image_tensor = load_and_preprocess_image(filepath, image_size=diffusion.model.resolution).to(diffusion.device)
        image_noise_levels = get_noise_level_values(timesteps, input_image_tensor)
        plt.plot(timesteps.cpu().numpy(), image_noise_levels.cpu().numpy(), color='red', label='Fake' if i == 0 else "")

    



    # Convert noise_levels to CPU for plotting if necessary
    plt.title('Average Predicted Noise Level Across Timesteps')
    plt.xlabel('Timestep')
    plt.ylabel('Average Predicted Noise Level')
    plt.savefig('foo.png')
    plt.show()
