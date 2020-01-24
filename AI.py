
# coding: utf-8

# In[8]:


get_ipython().system('git clone https://www.github.com/genekogan/neural-style-pt')
get_ipython().run_line_magic('cd', 'neural-style-pt')
get_ipython().system('git checkout modular')
get_ipython().system('pip install ninja')
get_ipython().system('python models/download_models.py')


# In[9]:


from model import *
from utils import *
import torch
import torchvision.transforms as transforms
from PIL import Image
import requests
from io import BytesIO
from imageio import imread


def optimize(stylenet, img, num_iterations, output_path, original_colors, print_iter=None, save_iter=None):
    def iterate():
        t[0] += 1
    
        optimizer.zero_grad()
        stylenet(img)
        loss = stylenet.get_loss()
        loss.backward()
        
        maybe_print(stylenet, t[0], print_iter, num_iterations, loss)
        maybe_save(img, t[0], save_iter, num_iterations, original_colors, output_path)

        return loss
    
    img = nn.Parameter(img.type(stylenet.dtype))
    optimizer, loopVal = setup_optimizer(img, stylenet.params, num_iterations)
    t = [0]
    while t[0] <= loopVal:
        optimizer.step(iterate)
    
    return img


def preprocess_from_url(image, image_size, to_normalize=True):
    if type(image_size) is not tuple:
        image_size = tuple([int((float(image_size) / max(image.size))*x) for x in (image.height, image.width)])
    Loader = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])
    rgb2bgr = transforms.Compose([transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])])])
    if to_normalize:
        Normalize = transforms.Compose([transforms.Normalize(mean=[103.939, 116.779, 123.68], std=[1,1,1])])
        tensor = Normalize(rgb2bgr(Loader(image) * 256)).unsqueeze(0)
    else:
        tensor = rgb2bgr(Loader(image)).unsqueeze(0)
    return tensor


def load_image_url(url, image_size, to_normalize=True):
  image = Image.fromarray(imread(url)).convert('RGB')
  return preprocess_from_url(image, image_size, to_normalize)


# setup stylenet
params = StylenetArgs()
params.gpu = '0'
params.backend = 'cudnn'

dtype, multidevice, backward_device = setup_gpu(params)
stylenet = StyleNet(params, dtype, multidevice, backward_device)


# In[ ]:


content_path = 'https://cdn.britannica.com/79/96379-050-FDCFF8D3/landmark-tower-Markaz-al-Mamlakah-Saudi-Arabia-Riyadh.jpg'
style_paths = ['https://i.pinimg.com/236x/77/40/20/774020bd7d17f21a2cb50a3e07f0a1ae.jpg',]

num_iterations = 1000
output_path = 'out.png'
image_size = 720
original_colors = 1
style_scale = 1.0
print_iter = 100
save_iter = 100


# load content image
content_image = load_image_url(content_path, image_size)

# load style images
style_size = int(image_size * style_scale)
style_images = [load_image_url(path, style_size) for path in style_paths]

# set hyper-parameters
stylenet.set_content_weight(5e0) # the most important 
stylenet.set_style_weight(1e2) # the most important c

stylenet.set_hist_weight(0)
stylenet.set_tv_weight(1e-3)
stylenet.set_style_statistic('gram')

# capture the style and content images
stylenet.capture(content_image, style_images)

# initialize with a random image
img = random_image_like(content_image)

# optimize!
img = optimize(stylenet, img, num_iterations, output_path, original_colors, print_iter, save_iter)

# display
deprocess(img)

