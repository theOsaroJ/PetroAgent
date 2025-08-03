import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
import joblib

MODEL_PATH = "nn_model.pt"

class SimpleNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_neural():
    data = load_iris(as_frame=True)
    X = torch.tensor(data.data.values, dtype=torch.float32)
    y = torch.tensor(data.target.values, dtype=torch.long)
    model = SimpleNet(input_dim=4, hidden_dim=16, output_dim=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(50):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
    torch.save(model.state_dict(), MODEL_PATH)
    return {"message": "neural net trained", "model_path": MODEL_PATH}

def predict_neural():
    data = load_iris(as_frame=True)
    sample = torch.tensor([5.1, 3.5, 1.4, 0.2], dtype=torch.float32)
    model = SimpleNet(input_dim=4, hidden_dim=16, output_dim=3)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    with torch.no_grad():
        logits = model(sample)
        pred = int(torch.argmax(logits).item())
    return {"prediction": pred}
