import torch
from models import Generator
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


def test_checkpoint(checkpoint_path, input_image_path, output_path=None):
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = Generator().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load and preprocess image
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    img = Image.open(input_image_path).convert('L')
    img_tensor = transform(img).unsqueeze(0).to(device)

    # Generate colorized version
    with torch.no_grad():
        output = model(img_tensor)

    # Convert output to image
    output = output.cpu().squeeze(0)
    output = output * 0.5 + 0.5  # Denormalize
    output = transforms.ToPILImage()(output)

    # Display results
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Input')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(output)
    plt.title('Colorized')
    plt.axis('off')

    if output_path:
        plt.savefig(output_path)
    plt.show()

    return output


# Example usage
if __name__ == "__main__":
    checkpoint_path = "checkpoints/checkpoint_epoch_100.pth"
    test_image_path = "processed_images/89707276_bw_processed.jpg"  # Your test image
    colorized = test_checkpoint(checkpoint_path, test_image_path)