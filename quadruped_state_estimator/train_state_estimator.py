import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd

# Define the MLP model
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.layer1 = nn.Linear(45, 256)
        self.layer2 = nn.Linear(256, 128)
        self.output_layer = nn.Linear(128, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.output_layer(x)
        return x

# Load dataset from CSV file
def load_dataset_from_csv(filepath):
    data = pd.read_csv(filepath)
    data = torch.tensor(data.values, dtype=torch.float32)
    inputs = data[:, 3:]  # The rest of the columns as inputs
    targets = data[:, :3]  # First 3 columns as targets
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
    model = SimpleMLP()

    # Load dataset from CSV
    filepath = "quadruped_state_estimator/dataset.csv"  # Replace with your CSV file path
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
    model_path = "quadruped_state_estimator/simple_mlp_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Load the model state from the file for inference
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f"Model loaded from {model_path}")

    # Example prediction
    input_tensor = torch.tensor([-0.24631839,0.18723287,-0.09318641,0.028049123,-0.025900668,-1.0330632,0.46058035,0.2479403,-0.30430126,0.034202546,0.1014897,-0.021568187,0.006815998,-0.00015507918,0.005955586,0.016301272,-0.007733185,0.1165862,-0.11805348,-0.076901406,0.12329475,2.7562032,5.732006,-1.7907379,1.2611842,-0.92328686,1.3570652,2.1867409,0.55202556,4.9861336,-6.368494,-3.5004716,5.1214876,1.0085579,2.206266,-0.30466098,0.07726368,-0.030805215,0.28187332,0.33519456,-0.3987426,2.664197,-2.5022304,-1.811091,2.9067085])  # Single input example
    with torch.no_grad():
        output_tensor = model(input_tensor)
