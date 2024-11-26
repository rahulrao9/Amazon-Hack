import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import ast

class BRAIN(nn.Module):
    def __init__(self, input_size=1536, hidden_size=2048, z_size=4, output_size=4):
        super(BRAIN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size * z_size, 8192)
        self.dropout = nn.Dropout(0.4)
        self.fc3 = nn.Linear(8192, output_size)

    def forward(self, embedding, z):
        x = self.fc1(embedding)
        # print("x shape:", x.unsqueeze(2).size())
        # print("z shape:", z.unsqueeze(1).size())
        x = torch.matmul(x.unsqueeze(2), z.unsqueeze(1))
        x = x.view(x.size(0), -1)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        return x

    def custom_backward(self, loss, z):
        loss.backward()
        # with torch.no_grad():
        #     grad_8192 = self.fc2.weight.grad
        #     z_inv = torch.pinverse(z)
        #     grad_2048 = torch.matmul(grad_8192.view(-1, z.size(1)), z_inv)
        #     self.fc1.weight.grad = grad_2048.view(self.fc1.weight.size())

class CustomDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        embedding = np.array(ast.literal_eval(self.data.iloc[idx]['EMBEDDING']), dtype=np.float32)
        z_list = np.array(ast.literal_eval(self.data.iloc[idx]['z_bar']), dtype=np.float32)
        output = np.array(ast.literal_eval(self.data.iloc[idx]['output']), dtype=np.float32)
        
        return torch.tensor(embedding), torch.tensor(z_list), torch.tensor(output)

def train(model, optimizer, criterion, train_loader, val_loader, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_idx, (embedding, z, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(embedding, z)
            loss = criterion(output, target)
            model.custom_backward(loss, z)
            optimizer.step()
            train_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}')
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for embedding, z, target in val_loader:
                output = model(embedding, z)
                loss = criterion(output, target)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f'Epoch {epoch}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()
    print(f"Model loaded from {path}")

# Example usage
input_size = 1536
hidden_size = 2048
z_size = 4
output_size = 4

model = BRAIN(input_size, hidden_size, z_size, output_size)
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

# Create datasets and data loaders
train_dataset = CustomDataset('train_800.csv')
test_dataset = CustomDataset('test_200.csv')

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Train the model
train(model, optimizer, criterion, train_loader, test_loader, num_epochs=10)

# # Save the model after training
# save_model(model, 'BRAIN_v1.pth')

# # To load the model later
# loaded_model = BRAIN(input_size, hidden_size, z_size, output_size)
# load_model(loaded_model, 'BRAIN_v1.pth')

# # Example inference with loaded model
# sample_data = next(iter(test_loader))
# sample_embedding, sample_z, _ = sample_data
# output = loaded_model(sample_embedding, sample_z)
# print(output.shape)  # Should be torch.Size([32, 4])

# # Evaluate the model on the test set
# loaded_model.eval()
# test_loss = 0
# correct = 0
# total = 0
# with torch.no_grad():
#     for embedding, z, target in test_loader:
#         output = loaded_model(embedding, z)
#         test_loss += criterion(output, target).item()
#         _, predicted = output.max(1)
#         total += target.size(0)
#         correct += predicted.eq(target.max(1)[1]).sum().item()

# avg_test_loss = test_loss / len(test_loader)
# accuracy = correct / total
# print(f'Test Loss: {avg_test_loss:.4f}, Accuracy: {accuracy:.4f}')