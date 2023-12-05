import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
# To test on individiaul images


save_model_path = ''
# Load and preprocess an individual image
def load_and_preprocess_image(image_path):
    transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize((48, 48)),
    transforms.Grayscale(),
    transforms.ToTensor(),
        F.normalize
    ])

    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

# Run the model on an individual image
def run_model_on_image(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

    return predicted_class, probabilities

# Example Usage:
if __name__ == "__main__":
    # Load the saved model
      # model.eval()
    model = torch.load(save_model_path + '/trained_model.pth')
    # Load and preprocess an individual image
    image_path = save_model_path + "/focusedimge.jpg"
    image_tensor = load_and_preprocess_image(image_path)

    # Run the model on the individual image
    predicted_class, probabilities = run_model_on_image(model, image_tensor)
    print(f"Predicted Class: {predicted_class}")
    print(f"Class Probabilities: {probabilities}")
