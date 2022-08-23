from torchvision import transforms
from PIL import Image
import cv2
import utils

images = utils.get_images("images")
#img = Image.open(r"images/damaged_cod/GOPR0317.JPG")
#img = cv2.imread("images/damaged_cod/GOPR0317.JPG")
#img.show()

for i in range(0,len(images),5):
    print(i)
    img = cv2.imread(images[i])

    transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.CenterCrop(4000),
                transforms.Resize(448)                           
                #transforms.ToTensor()                              
                ])
    img_transformed = transform(img)

    img_transformed.show()
