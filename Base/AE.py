#图像 → 拉直 → 压缩（encoder） → 还原（decoder） → 变回图像

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

#准备数据
# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Download and load the MNIST dataset
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

#构建自编码模型
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)  # Latent space dimension
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid()  # Output values between 0 and 1
        )
    
    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the input
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded.view(-1, 1, 28, 28)  # Reshape back to image format


#训练自编码器
# Initialize the model, loss function, and optimizer
model = Autoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
num_epochs = 10
outputs = []

for epoch in range(num_epochs):
    total_loss = 0
    for data in train_loader:
        img, _ = data
        
        # Forward pass
        output = model(img)
        loss = criterion(output, img)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    # Print loss at each epoch
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
    
    # Save the output images
    outputs.append((epoch, img, output))

print("Training complete!")

#结果可视化
def plot_reconstruction(epoch, original, reconstructed):
    plt.figure(figsize=(10, 4))
    
    # Plot original images
    for i in range(5):
        plt.subplot(2, 5, i+1)
        plt.imshow(original[i][0].detach().numpy(), cmap='gray')
        plt.title("Original")
        plt.axis('off')
        
    # Plot reconstructed images
    for i in range(5):
        plt.subplot(2, 5, i+6)
        plt.imshow(reconstructed[i][0].detach().numpy(), cmap='gray')
        plt.title("Reconstructed")
        plt.axis('off')
        
    plt.suptitle(f"Epoch {epoch+1}")
    plt.tight_layout()
    plt.show()

# Visualize the last epoch result
epoch, imgs, outputs = outputs[-1]
plot_reconstruction(epoch, imgs, outputs)











