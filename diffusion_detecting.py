import torch
from torchvision import transforms
from datasets import load_dataset
from torch.utils.data import DataLoader
from diffusers import UNet2DModel, DDPMScheduler
from PIL import Image


# Load the dataset
fashion_mnist = load_dataset("fashion_mnist")

# Adjust transformation to ensure compatibility with UNet
preprocess = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize to 32x32
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Transform function to apply preprocessing
def transform(examples):
  images = [preprocess(image.convert("L")) for image in
  examples["image"]]
  return {"images": images, "labels": examples["label"]}

# Apply transform to the dataset
train_dataset = fashion_mnist["train"].with_transform(transform)
batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize the model
model = UNet2DModel(
    in_channels=1,  # 1 channel for grayscale images
    out_channels=1,  # output channels must also be 1
    sample_size=32,
    block_out_channels=(32, 64, 128, 256),
    norm_num_groups=8,
    num_class_embeds=10,  # Enable class conditioning
)

# Setup for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

# Initialize the scheduler
scheduler = DDPMScheduler(num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02)

# Function to show images (optional for debugging)
def show_images(images):
    to_pil = ToPILImage()
    for img in images:
        display(to_pil(img))

# Training loop
for step, batch in enumerate(train_dataloader):
    clean_images = batch["images"].to(device)
    class_labels = batch["labels"].to(device)

    # Sample noise
    noise = torch.randn_like(clean_images)

    # Sample a random timestep for each image
    timesteps = torch.randint(0, 1000, (batch_size,), device=device)

    # Add noise to the clean images according to the timestep
    noisy_images = scheduler.add_noise(clean_images, noise, timesteps)

    # Get the model prediction for the noise
    noise_pred = model(noisy_images, timesteps, class_labels=class_labels, return_dict=False)[0]
    # Calculate the loss and update the parameters
    optimizer.zero_grad()
    loss = loss_fn(noise_pred, noisy_images)  # Adjust as needed to match your model's output
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(f"Step {step}: Loss {loss.item()}")

    if step == 1100:
        break

# Assuming 'model' is already defined and trained
# Save just the model parameters or state dictionary
torch.save(model.state_dict(), 'model_state_dict.pth')

# Optionally, save the entire model
torch.save(model, 'entire_model.pth')

