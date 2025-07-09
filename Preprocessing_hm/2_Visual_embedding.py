# +
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
from PIL import ImageFile
import matplotlib.pyplot as plt
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.LOAD_TRUNCATED_IMAGES = True

import torchvision.transforms as transforms
import torchvision.models as models

import copy
from tqdm import tqdm
import pickle
import csv


# +
###What is the dataset name?###
dname = "hm"

###What is the Dataset folder path?###
dataset_path = '../Dataset'

###What is the Image folder path?###
image_dir = f'{dataset_path}/{dname}/Images'

###What is the ProductDictName path?###
#ProductDictFile = f'{dataset_path}/{dname}/Product/{dname}_productdetail'

###What is the IndexDictionary path?###
indexDictionaryFile = f'{dataset_path}/{dname}/Product/idtoindex_trousers_{dname}.csv'


###Image Size?###
imsize = 512

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# -

# ### Image Loader

# +
loader = transforms.Compose([
    transforms.Resize(imsize),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor

def image_loader(image_name, device):
    image = Image.open(image_name)
    image = image.convert("RGB")
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


# -

# ### Gram Matrix

def gram_matrix(input, kernel_size, stride):
    max_pool = torch.nn.MaxPool2d(kernel_size = kernel_size, stride = stride)
    avg_pool = torch.nn.AvgPool2d(kernel_size = kernel_size, stride = stride)
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)

    gram = torch.mm(features, features.t())
    gram_normalized = gram.div(a * b * c * d)

    gram_dims = gram_normalized.size()
    gram_normalized = gram_normalized.view(1, gram_dims[0], gram_dims[1])

    gram_final = max_pool(gram_normalized)
    gram_final_dims = gram_final.size()

    return gram_final.view(1, gram_final_dims[1] * gram_final_dims[2])


# ### Normalization

# create a module to normalize input image so we can easily put it in a
# nn.Sequential
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


# ### GET GRAM MATRIX

# +
# desired depth layers to compute style/content losses :

def get_gram_matrixes(cnn, normalization_mean, normalization_std, style_img, style_layers):
    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(device)
    
    gram_matrixes = []
    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in style_layers:
            if name == 'conv_1':
                gram_matrixes.append(gram_matrix(model(style_img).detach(), kernel_size = 4, stride = 4))
            elif name == 'conv_2':
                gram_matrixes.append(gram_matrix(model(style_img).detach(), kernel_size = 4, stride = 4))
            '''
            elif name == 'conv_3':
                gram_matrixes.append(gram_matrix(model(style_img).detach(), kernel_size = 16, stride = 16))
            elif name == 'conv_4':
                gram_matrixes.append(gram_matrix(model(style_img).detach(), kernel_size = 16, stride = 16))
            elif name == 'conv_5':
                gram_matrixes.append(gram_matrix(model(style_img).detach(), kernel_size = 32, stride = 32))
            if name == 'conv_6':
                gram_matrixes.append(gram_matrix(model(style_img).detach(), kernel_size = 32, stride = 32))
            elif name == 'conv_7':
                gram_matrixes.append(gram_matrix(model(style_img).detach(), kernel_size = 32, stride = 32))
            elif name == 'conv_8':
                gram_matrixes.append(gram_matrix(model(style_img).detach(), kernel_size = 32, stride = 32))
            
            elif name == 'conv_9':
                gram_matrixes.append(gram_matrix(model(style_img).detach(), kernel_size = 64, stride = 64))
            elif name == 'conv_10':
                gram_matrixes.append(gram_matrix(model(style_img).detach(), kernel_size = 64, stride = 64))
            if name == 'conv_11':
                gram_matrixes.append(gram_matrix(model(style_img).detach(), kernel_size = 64, stride = 64))
            elif name == 'conv_12':
                gram_matrixes.append(gram_matrix(model(style_img).detach(), kernel_size = 64, stride = 64))
            elif name == 'conv_13':
                gram_matrixes.append(gram_matrix(model(style_img).detach(), kernel_size = 64, stride = 64))
            elif name == 'conv_14':
                gram_matrixes.append(gram_matrix(model(style_img).detach(), kernel_size = 64, stride = 64))
            elif name == 'conv_15':
                gram_matrixes.append(gram_matrix(model(style_img).detach(), kernel_size = 64, stride = 64))
            elif name == 'conv_16':
                gram_matrixes.append(gram_matrix(model(style_img).detach(), kernel_size = 64, stride = 64))

            '''
    return gram_matrixes


# -

def ImageFeature(productName, indexDictionary):
    
    #style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5', 'conv_6', 'conv_7', 'conv_8', 'conv_9', 'conv_10', 'conv_11', 'conv_12', 'conv_13', 'conv_14', 'conv_15', 'conv_16']
    style_layers = ['conv_1', 'conv_2']
    
    cnn = models.vgg19(pretrained=True).features.to(device).eval()
    normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    

    gram_matrix_dict = {i:[] for i in range(1,len(productNames)+1)}
    
    for l in tqdm(range(len(productNames))):
        my_image = image_loader(f'{image_dir}/{productNames[l]}.jpg', device)
        gram_matrix_of_an_image = get_gram_matrixes(cnn, normalization_mean, normalization_std, my_image, style_layers)
        gram_matrix_dict[indexDictionary[productNames[l]]] = torch.cat((gram_matrix_of_an_image[0], gram_matrix_of_an_image[1]), 1)[0]
        #gramFinal = torch.cat((gram_matrixes[l][0], gram_matrixes[l][1], gram_matrixes[l][2], gram_matrixes[l][3], gram_matrixes[l][4], gram_matrixes[l][5], gram_matrixes[l][6], gram_matrixes[l][7], gram_matrixes[l][8], gram_matrixes[l][9], gram_matrixes[l][10], gram_matrixes[l][11], gram_matrixes[l][12], gram_matrixes[l][13], gram_matrixes[l][14], gram_matrixes[l][15]), 1)

    #xxxxxxxx
    #gram_matrix_dict[indexDictionary['xxxxxxxx']] = torch.zeros(imsize)
    return gram_matrix_dict


# +
##Product Names

# file = open(ProductDictFile, 'rb')
# productDictFinal = pickle.load(file)
# productNames = list(productDictFinal[i][0] for i in range(len(productDictFinal)))

# print(len(productDictFinal))
# print(len(productNames))

# +
## Product Index 
with open(indexDictionaryFile, newline='') as f:
    reader = csv.reader(f)
    idtoindex = list(reader)

indexDictionary = {}
productNames = []

for l in range(len(idtoindex)):
    productNames.append(str(idtoindex[l][0]))
    indexDictionary[str(idtoindex[l][0])] = int(idtoindex[l][1])
print(len(indexDictionary))
print(len(productNames))
# -

gram_matrixes_final = ImageFeature(productNames, indexDictionary)
print(len(gram_matrixes_final))

gram_matrixes = torch.zeros([len(gram_matrixes_final) + 1, imsize])
gram_matrixes[0] = torch.zeros(imsize).to(device)
for l in tqdm(range(1,len(gram_matrixes_final))):
    gram_matrixes[l] = gram_matrixes_final[l].to(device)
print(gram_matrixes.size())

file = open(f'{dataset_path}/{dname}/gram_matrixes_maxPool_first2_{imsize}_trousers', 'wb')
pickle.dump(gram_matrixes, file)
file.close()


