def restore_model(resume_iters):
    model_save_dir = r"C:\Users\YongTaek\Desktop\stargan_both\models"
    """Restore the trained generator and discriminator."""
    print('Loading the trained models from step {}...'.format(resume_iters))
    G_path = os.path.join(model_save_dir, '{}-G.ckpt'.format(resume_iters))
    D_path = os.path.join(model_save_dir, '{}-D.ckpt'.format(resume_iters))
    G.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
    D.load_state_dict(torch.load(D_path, map_location=lambda storage, loc: storage))
    
def build_model():
    from model import Generator
    from model import Discriminator
    """Create a generator and a discriminator."""
    if dataset in ['CelebA', 'RaFD']:
        G = Generator(g_conv_dim, c_dim, g_repeat_num)
        D = Discriminator(image_size, d_conv_dim, c_dim, d_repeat_num) 
    elif dataset in ['Both']:
        G = Generator(g_conv_dim, c_dim+c2_dim+2, g_repeat_num)   # 2 for mask vector.
        D = Discriminator(image_size, d_conv_dim, c_dim+c2_dim, d_repeat_num)

    g_optimizer = torch.optim.Adam(G.parameters(), g_lr, [beta1, beta2])
    d_optimizer = torch.optim.Adam(D.parameters(), d_lr, [beta1, beta2])
    print_network(G, 'G')
    print_network(D, 'D')

    G.to(device)
    D.to(device)
    
def print_network(model, name):
    """Print out the network information."""
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(model)
    print(name)
    print("The number of parameters: {}".format(num_params))
    
def get_loader(image_dir, attr_path, selected_attrs, crop_size=178, image_size=128, 
               batch_size=16, dataset='CelebA', mode='train', num_workers=1):
    """Build and return a data loader."""
    transform = []
    if mode == 'train':
        transform.append(T.RandomHorizontalFlip())
    transform.append(T.CenterCrop(crop_size))
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    if dataset == 'CelebA':
        dataset = CelebA(image_dir, attr_path, selected_attrs, transform, mode)
    elif dataset == 'RaFD':
        dataset = ImageFolder(image_dir, transform)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers)
    return data_loader

def create_labels(c_org, c_dim=5, dataset='CelebA', selected_attrs=None):
    """Generate target domain labels for debugging and testing."""
    # Get hair color indices.
    if dataset == 'CelebA':
        hair_color_indices = []
        for i, attr_name in enumerate(selected_attrs):
            if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                hair_color_indices.append(i)

    c_trg_list = []
    for i in range(c_dim):
        if dataset == 'CelebA':
            c_trg = c_org.clone()
            if i in hair_color_indices:  # Set one hair color to 1 and the rest to 0.
                c_trg[:, i] = 1
                for j in hair_color_indices:
                    if j != i:
                        c_trg[:, j] = 0
            else:
                c_trg[:, i] = (c_trg[:, i] == 0)  # Reverse attribute value.
        elif dataset == 'RaFD':
            c_trg = label2onehot(torch.ones(c_org.size(0))*i, c_dim)

        c_trg_list.append(c_trg.to(device))
    return c_trg_list

def label2onehot(labels, dim):
    """Convert label indices to one-hot vectors."""
    batch_size = labels.size(0)
    out = torch.zeros(batch_size, dim)
    out[np.arange(batch_size), labels.long()] = 1
    return out

def denorm(x):
    """Convert the range from [-1, 1] to [0, 1]."""
    out = (x + 1) / 2
    return out.clamp_(0, 1)


import sys
sys.path.append(r"C:\Users\YongTaek\Desktop\GAN\StarGAN-master")


from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random

dataset="Both"
d_conv_dim = 64
g_conv_dim = 64
c_dim = 5
c2_dim = 8
g_repeat_num = 6
d_repeat_num = 6
g_lr = 0.0001
d_lr = 0.0001
beta1 = 0.5
beta2 = 0.999
image_size = 256
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
crop_size=256

from model import Generator
from model import Discriminator
"""Create a generator and a discriminator."""
if dataset in ['CelebA', 'RaFD']:
    G = Generator(g_conv_dim, c_dim, g_repeat_num)
    D = Discriminator(image_size, d_conv_dim, c_dim, d_repeat_num) 
elif dataset in ['Both']:
    G = Generator(g_conv_dim, c_dim+c2_dim+2, g_repeat_num)   # 2 for mask vector.
    D = Discriminator(image_size, d_conv_dim, c_dim+c2_dim, d_repeat_num)

g_optimizer = torch.optim.Adam(G.parameters(), g_lr, [beta1, beta2])
d_optimizer = torch.optim.Adam(D.parameters(), d_lr, [beta1, beta2])
print_network(G, 'G')
print_network(D, 'D')

G.to(device)
D.to(device)

#rafd_image_dir = r'C:\Users\YongTaek\Desktop\test'
rafd_crop_size = 256
batch_size = 16
mode = 'test'
num_workers = 1

selected_attrs = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']
import numpy as np
from torchvision.utils import save_image

result_dir = r'C:\Users\YongTaek\Desktop\practice'
transform = []
#transform.append(T.CenterCrop(256))
transform.append(T.CenterCrop(crop_size))
transform.append(T.Resize(image_size))
transform.append(T.ToTensor())
transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
transform = T.Compose(transform)

def test_multi(image_dir, test_iters=190000, result_dir=result_dir, c_org1=[1,0,0,1,1], c_org2=[0,0,0,0,0,1,0,0]):
    # c_org1 = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']
    # c_org2 = angry, contemptuous, disgusted, fearful, happy, neutral, sad, surprised
    """Translate images using StarGAN trained on multiple datasets."""
    # Load the trained generator.
    restore_model(test_iters)
    x_real = Image.open(image_dir)
    x_real = x_real.resize((256,256))
    x_real = transform(x_real).unsqueeze(0)
    restore_model(test_iters)
    
    c_org1=torch.FloatTensor([c_org1])
    c_org2=torch.FloatTensor([c_org2])
    
    with torch.no_grad():
            # Prepare input images and target domain labels.
        x_real = x_real.to(device)
        c_celeba_list = create_labels(c_org1, c_dim, 'CelebA', selected_attrs)
        c_rafd_list = create_labels(c_org2, c2_dim, 'RaFD')
        zero_celeba = torch.zeros(x_real.size(0), c_dim).to(device)            # Zero vector for CelebA.
        zero_rafd = torch.zeros(x_real.size(0), c2_dim).to(device)             # Zero vector for RaFD.
        mask_celeba = label2onehot(torch.zeros(x_real.size(0)), 2).to(device)  # Mask vector: [1, 0].
        mask_rafd = label2onehot(torch.ones(x_real.size(0)), 2).to(device)     # Mask vector: [0, 1].

        # Translate images.
        x_fake_list = [x_real]
        for c_celeba in c_celeba_list:
            c_trg = torch.cat([c_celeba, zero_rafd, mask_celeba], dim=1)
            x_fake_list.append(G(x_real, c_trg))
        for c_rafd in c_rafd_list:
            c_trg = torch.cat([zero_celeba, c_rafd, mask_rafd], dim=1)
            x_fake_list.append(G(x_real, c_trg))

        # Save the translated images.
        x_concat = torch.cat(x_fake_list, dim=3)
        result_dir = os.path.join(result_dir, os.path.split(image_dir)[-1])
        save_image(denorm(x_concat.data.cpu()), result_dir, nrow=1, padding=0)
        print('Saved real and fake images into {}...'.format(result_dir))

