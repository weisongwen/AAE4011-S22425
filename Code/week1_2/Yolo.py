# https://colab.research.google.com/
from IPython import get_ipython
from IPython.display import display
# %%
!pip install torch torchvision torchaudio
!pip install yolov5
# %%
import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
from PIL import Image
import matplotlib.pyplot as plt

# Replace with the correct path to your image within the Colab environment
img_path = '/content/0000001_04527_d_0000008.jpg' 
img = Image.open(img_path)
plt.imshow(img)
plt.axis('off')
plt.show()
results = model(img)

results.show()
results.save('/content/pre_0000001_04527_d_0000008.jpg')