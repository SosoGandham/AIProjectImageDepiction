import torch
import torch.nn as nn  # For building the neural network
import torch.optim as optim  # For optimization algorithms
from torchvision import datasets, transforms  # For dataset handling and image transformations
from torch.utils.data import DataLoader, random_split  # For data loading and splitting
import matplotlib.pyplot as plt  # For visualizing results
from PIL import Image  # For loading test images
import os
print("Current Working Directory:", os.getcwd())

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set up transformations for dataset
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize all images to 128x128
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize pixel values
])

# Load the training and testing datasets
train_dataset = datasets.ImageFolder('archive/training_set', transform=transform)
test_dataset = datasets.ImageFolder('archive/test_set', transform=transform)
# Debug: Verify folder paths

print("Current Working Directory:", os.getcwd())
print("Does training_set exist?", os.path.exists('archive/training_set'))
print("Does test_set exist?", os.path.exists('archive/test_set'))


# Debug: Check dataset class mappings and sizes
print("Class mappings:", train_dataset.class_to_idx)  # Should output: {'cats': 0, 'dogs': 1}
print("Number of training images:", len(train_dataset))  # Prints the number of training images
print("Number of testing images:", len(test_dataset))  # Prints the number of testing images

# Split the training dataset into training (80%) and validation (20%)
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

# Debug: Check sizes of training and validation splits
print("Number of training images after split:", len(train_dataset))
print("Number of validation images after split:", len(val_dataset))

# Create data loaders for training, validation, and testing
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # Shuffle for randomness
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  # No shuffle for validation
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # No shuffle for testing

# Define the CNN Model
class CatDogCNN(nn.Module):  # Define the custom CNN class
    def __init__(self):
        super(CatDogCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)  # First convolutional layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Pooling layer
        self.fc1 = nn.Linear(32 * 64 * 64, 2)  # Fully connected layer

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # Apply convolution, activation, and pooling
        x = x.view(-1, 32 * 64 * 64)  # Flatten the tensor for the fully connected layer
        x = self.fc1(x)  # Pass through the fully connected layer
        return x

# Initialize the model
model = CatDogCNN().to(device)  # Move model to device (GPU or CPU)

# Define the loss function (CrossEntropyLoss for classification)
criterion = nn.CrossEntropyLoss()

# Define the optimizer (SGD with learning rate and momentum)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Number of epochs
epochs = 3

# Lists to track training and validation loss
train_losses, val_losses = [], []

# Training and validation loop
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    model.train()  # Set the model to training mode
    running_loss = 0.0  # Track the cumulative training loss
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)  # Move data to device

        optimizer.zero_grad()  # Reset gradients
        outputs = model(images)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
        running_loss += loss.item()  # Accumulate loss

        # Print progress every 10 batches
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

    train_losses.append(running_loss / len(train_loader))  # Average loss for this epoch

    # Validation phase
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0  # Track validation loss
    correct = 0  # Count correct predictions
    total = 0  # Total images
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)  # Move data to device
            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            val_loss += loss.item()  # Accumulate loss
            _, predicted = torch.max(outputs, 1)  # Get predicted class
            total += labels.size(0)  # Increment total count
            correct += (predicted == labels).sum().item()  # Count correct predictions

    val_losses.append(val_loss / len(val_loader))  # Average validation loss
    accuracy = 100 * correct / total  # Calculate accuracy
    print(f"  Validation Loss: {val_losses[-1]:.4f}, Accuracy: {accuracy:.2f}%")

# Save the trained model
torch.save(model.state_dict(), 'cat_dog_model.pth')

# Plot training and validation loss
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Load a new image and make a prediction
image_path = 'test_set/cats/cat1.jpg'  # Replace with a valid test image path
try:
    image = Image.open(image_path)
except FileNotFoundError:
    print(f"Error: The file at {image_path} was not found. Please check the file path.")
    exit()

# Apply transformations and predict
tensor_image = transform(image).unsqueeze(0).to(device)  # Apply transformations and move to device
model.load_state_dict(torch.load('cat_dog_model.pth'))  # Load the trained model
model.eval()  # Set the model to evaluation mode
output = model(tensor_image)
_, predicted_class = torch.max(output, 1)
print(f'Predicted class: {predicted_class.item()}')  # Print the class index
