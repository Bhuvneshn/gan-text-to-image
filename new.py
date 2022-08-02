import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from txt2image_dataset import Text2ImageDataset
import yaml
import torchvision
from models.gan_factory import gan_factory
import os
  

generator=gan_factory.generator_factory("gan").cuda()

gen = torch.load("checkpoints\gen_190.pth")
state_dict = {}

for key in gen.keys():
    new_key = key.strip("module.")
    print(new_key)
    state_dict[new_key] = gen[key]
    
    
generator.load_state_dict(state_dict,strict=False)
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)


dataset = Text2ImageDataset(config['flowers_dataset_path'], split=0)
data_loader = DataLoader(dataset, batch_size=64, shuffle=True,
                         num_workers=0)
i=1
for sample in data_loader:
    right_images = sample['right_images']
    right_embed = sample['right_embed']
    wrong_images = sample['wrong_images']
    

    right_images = Variable(right_images.float()).cuda()
    right_embed = Variable(right_embed.float()).cuda()
    wrong_images = Variable(wrong_images.float()).cuda()
    noise = Variable(torch.randn(64, 100), volatile=True).cuda()
    noise = noise.view(noise.size(0), 100, 1, 1)
    fake_images = generator(right_embed, noise)
    print(fake_images.shape)
    directory = "img{}".format(i)
    parent_dir = "C:/Python Projs/Text-to-Image-Synthesis-master (1)/Text-to-Image-Synthesis-master/visualize_images"
    path = os.path.join(parent_dir, directory)
    os.mkdir(path)
    torchvision.utils.save_image(fake_images.data, str(parent_dir)+'/'+str(directory)+'/'+'img{}.png'.format(i))
    with open(str(parent_dir)+'/'+str(directory)+'/'+'text.txt', 'w') as f:
        for line in list(sample['txt']):
            f.write(f"{line}\n")
    i+=1



  


  

