import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd

# Define the MLP model
class StateEstimator(nn.Module):
    def __init__(self):
        super(StateEstimator, self).__init__()
        # 60 - 6 = 54
        self.layer1 = nn.Linear(42, 256)
        self.layer2 = nn.Linear(256, 128)
        self.output_layer = nn.Linear(128, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.output_layer(x)
        return x

def load_dataset_from_csv(filepath):
    data = pd.read_csv(filepath)
    data = torch.tensor(data.values, dtype=torch.float32)
    # Select columns 3 to 8 and 12 onwards for inputs
    # note: we don't want to include velocity commands since predictions shouldn't really be based on user inputs
    inputs = torch.cat([data[:, 3:9], data[:, 12:]], dim=1)  
    # First 3 columns as targets
    targets = data[:, :3]  
    return inputs, targets


# Training function
def train(model, dataloader, criterion, optimizer, epochs=20):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(dataloader):.4f}")

# Evaluation function
def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(dataloader)

# Main function
if __name__ == "__main__":
    # Create the model
    model = StateEstimator()

    # Load dataset from CSV
    filepath = "/home/fumi/CodeStuff/IsaacLab/dataset.csv"  # Replace with your CSV file path
    inputs, targets = load_dataset_from_csv(filepath)

    # Create TensorDataset
    dataset = TensorDataset(inputs, targets)

    # Split dataset into training and testing sets
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train(model, train_dataloader, criterion, optimizer, epochs=50)

    # Evaluate the model
    test_loss = evaluate(model, test_dataloader, criterion)
    print(f"Test Loss: {test_loss:.4f}")

    # Save the model to a file
    model_path = "quadruped_state_estimator/state_estimator.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Load the model state from the file for inference
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f"Model loaded from {model_path}")

    # Export to ONNX
    onnx_path = "quadruped_state_estimator/state_estimator.onnx"
    dummy_input = torch.randn(1, 42)  # Create a dummy input tensor with the shape of (1, 45)
    torch.onnx.export(model, dummy_input, onnx_path, input_names=['input'], output_names=['output'])
    print(f"Model exported to ONNX format at {onnx_path}")
