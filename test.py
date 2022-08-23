import torch 

resnet18 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
modules = list(resnet18.children())[:-1]
resnet18 = torch.nn.Sequential(*modules)
resnet18.eval()

import urllib
url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)

filename = "images/young_cod/GOPR0014.JPG"

from PIL import Image
from torchvision import transforms
input_image = Image.open(filename)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the resnet18

# move the input and resnet18 to GPU for speed if available
if torch.cuda.is_available():
    print("CUDA is available")
    input_batch = input_batch.to('cuda')
    resnet18.to('cuda')

with torch.no_grad():
    output = resnet18(input_batch)
# Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
#print(output[0])
print(output)

# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
#probabilities = torch.nn.functional.softmax(output[0], dim=0)
#print(probabilities)

