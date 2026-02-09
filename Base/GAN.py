#pip install torch torchvision numpy matplotlib

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
form torch.utils.data import DataLoader

#Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

#加载MNIST手写数据集
#Define image transformations
transform = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.5],std=[0.5])    #Normalize to [-1,1]
])

#Load MNIST dataset
train_dataset = torchvision.datasets.MNIST(
  root = './data',
  train = True,
  download = True,
  transform = transform
)

#Create data Loader
train_loader = DataLoader(
  train_dataset,
  batch_size = batch_size,
  shuffle = True
)

#构建判别器
#Deep Convolutional GAN
class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator, self).__init__()  #调用父类nn.Module的初始化，让模型注册参数

    #A simple network with convolutional layers
    self.model = nn.Sequential(   #按顺序一层一层往下写
      #Input is 1x28x28
      nn.Conv2d(1, 32, kernal_size=4, stride = 2, padding = 1),#从一个通道（灰度）提取32个特征，同时下采样，并防止边缘丢太多信息
      nn.LeakyReLU(0.2, inplace = True)  #x>0:x, x<0:0.2*x 防止死亡ReLU
      #32x14x14
      nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), #学习64个特征，再次下采样
      nn.BatchNorm2d(64),  #对每个通道标准化，让训练更稳定、更快收敛
      nn.LeakyReLU(0.2, inplace=True),
      # 64x7x7
      nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0),
      nn.BatchNorm2d(128),
      nn.LeakyReLU(0.2, inplace=True),
      # 128x3x3
      nn.Flatten(),   #压平（batch_size, 1152）
      nn.Linear(128 * 3 * 3, 1),  #全连接，把1152维特征映射成1个标量
      nn.Sigmoid()  # Output is probability between 0 and 1
    )
  def forward(self, x):
    return self.model(x)

#构建生成器
class Generator(nn.Module):
  def __init__(self, latent_dim):
    super(Generator, self).__init__()

    self.latent_dim = latent_dim       #噪声向量z的维度 如64/100/128

    #Initial size based on upsampling form random noise
    self.init_size = 7
    self.l1 = nn.Sequential(
      nn.Linear(latent_dim, 128 * self.init_size * self.init_size)
    )  #把一维噪声z投影成一个“低分辨率+多通道的特征图"
    #Upsample to 28x28 through convolutional layers
    self.model = nn.Sequatial(
      nn.BatchNorm2(128),
      #128x7x7
      nn.Upsample(scale_factor=2),
      nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(128),
      nn.LeakyReLU(0.2, inplace=True),
      # 128x14x14
      nn.Upsample(scale_factor=2),
      nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(64),
      nn.LeakyReLU(0.2, inplace=True),
      # 64x28x28
      nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
      nn.Tanh()  # Output normalized between -1 and 1
    )
  def forward(self, z):
    #Project and reshape the noise
    out = self.l1(z)
    out = out.view(out.shape[0], 128, self.init_size, self.init_size)
    #Generate the image
    
    img = self.model(out)
    return img

#模型初始化与优化器
#Hyperparameters
latent_dim = 100 #Dimension of the random noise
lr = 0.0002
beta1 = 0.5
beta2 = 0.999

#Initalize models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator(latent_dim).to(device)
discriminator = Discriminator().to(device)

#Binary cross entropy loss
adversarial_loss = nn.BECLoss()

#Oprimizers
optimizer_G = optim.Adam(generator.parameters(), lr = lr, betas = (beta1, beta2))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas = (beta1,beta2))


#可视化图像
def show_images(images, num_images = 25):
  #Convert tensor images to numpy images
  images = images.detach().cpu().numpy()
  #Rescale from [-1,1] to [0,1]
  images = (images + 1) / 2.0
  #Create a grid
  n = int(np.sqrt(num_images))
  plt.figure(figsize = (8,8))
  for i in range(num_images):
    plt.subplot(n , n , i + 1)\
    plt.imshow(images[i,0], cmap= 'gray')
    plt.axis('off')
  plt.tight_layout()
  plt.show()

#训练GAN
#Training parameters
num_epochs = 50
sample_interval = 10
samples_to_generate = 25

fixed_noise = torch.randn(samples_to_generate, latent_dim, device = device)

#Traning Loop
for epoch in range(num_epochs):
  for batch_idx, (real_imgs, _) in enumerate(train_loader):
    #move data to device
    real_imgs = real_imgs.to(device)
    batch_size = real_imgs.size(0)

  #labesl for real and fake imges
    real_label = torch.ones(batch_size, 1, device=device)
    fake_label = torch.zeros(batch_size, 1, device= device)
    
    # ---------------------
    # Train Discriminator
    # ---------------------
    optimizer_D.zero_grad()
    # Loss on real images
    real_pred = discriminator(real_imgs)
    d_loss_real = adversarial_loss(real_pred, real_label)

    #Loss on fake images
    z = torch.randn(batch_size, latent_dim, device=device)
    fake_imgs = generator(z)
    fake_pred = discriminator(fake_imgs.detach())
    d_loss_fake = adversarial_loss(fake_pred, fake_label)

    # Total discriminator loss
    d_loss = (d_loss_real + d_loss_fake) / 2
    
    d_loss.backward()
    optimizer_D.step()

    # ---------------------
    # Train Generator
    # ---------------------
    optimizer_G.zero_grad()
    
    # Generate new fake images
    fake_imgs = generator(z)
    fake_pred = discriminator(fake_imgs)
    
    # Try to fool the discriminator
    g_loss = adversarial_loss(fake_pred, real_label)
    
    g_loss.backward()
    optimizer_G.step()

    # Print progress
    if batch_idx % 100 == 0:
        print(
            f"[Epoch {epoch}/{num_epochs}] "
            f"[Batch {batch_idx}/{len(train_loader)}] "
            f"[D loss: {d_loss.item():.4f}] "
            f"[G loss: {g_loss.item():.4f}]"
        )


    # Generate and show example images
    if (epoch + 1) % sample_interval == 0:
        print(f"Generating sample images for epoch {epoch+1}...")
        generator.eval()
        with torch.no_grad():
            gen_imgs = generator(fixed_noise)
            show_images(gen_imgs, samples_to_generate)
        generator.train()
