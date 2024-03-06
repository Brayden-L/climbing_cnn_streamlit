import pandas as pd
import torch
from torch import nn
from torchvision import models
from torchvision.transforms import transforms
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
from natsort import natsorted

help_text = """
This is a demonstration of a convolutional neural network I trained to classify climbing images.
The idea is to classify climbing images, so to group images into topos, butt-shots and glory-shots. See images below for examples.
- Grouping topos is useful to a climber attempting to gather information about a climb or area.
- Grouping butt-shots is useful as they are typically regarded as lower quality images.
- Grouping glory-shots can be useful to automatically assign default images for routes or areas.

You may select from example images or upload your own.

I used a re-trained ImageNet pre-trained VGG11 model. Hyper-parameter optimization was done using Weights and Biases sweeps, with most gains coming from learning rate scheduling and light use of weight decay.

You can read an in-depth report [here](https://brayden-l.github.io/).
"""

def list_images_in_folder(folder_path):
    # Define a set of file extensions to consider as images
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    # Use pathlib to list all files in the folder
    folder = Path(folder_path)
    # Filter files by checking the suffix against the set of image extensions
    image_files = [file for file in folder.iterdir() if file.suffix.lower() in image_extensions]
    # Convert the Path objects to strings (if needed) and return the list
    return natsorted([str(image) for image in image_files])

img_captions = {'1.jpg': 'Typical glory-shot, in profile with a landscape background.',
                '2.jpg': 'Another typical glory-shot, with a harsh downward angle focusing on the climber\'s expression.',
                '3.jpg': 'Typical butt-shot, directly under the climber.',
                '4.jpg': 'Another typical butt-shot, one of the more exaggerated I have seen',
                '5.jpg': 'Typical topo, overlay style.',
                '6.jpg': 'Another typical topo, hand-drawn style.',
                '7.jpg': 'An example where the model is not sure if it is a glory-shot or butt-shot, and neither am I, but I lean toward glory-shot.',
                '8.jpg': 'An example where it is technically not a butt-shot, but the angle is so reminiscent of one that the model thinks it is a butt-shot.',
                '9.jpg': 'A straight-on and distant shot. I was worried these may be misclassified, but the model is robust to these.',
                '10.jpg': 'Feels like it could be close to being a butt-shot, but the landscape in the background is reminiscent of a glory-shot.',
                '11.jpg': 'Another that feels close to being a butt-shot, but the model is robust enough to correctly classify.',
                '12.jpg': 'Another typical landscape glory-shot',
                '13.jpg': 'A common down-angle glory-shot, highlighting the climber\'s span.',
                '14.jpg': 'A higher quality butt-shot, but a butt-shot nonetheless.',
                '15.jpg': 'Another distant glory-shot.',
                '16.jpg': 'Another basic overlay topo.',
                '17.jpg': 'A beautiful landscape glory-shot.',
                }

def resize_image_object_to_height(img, new_height):
    """
    Resize an already opened image to a specified height, maintaining the aspect ratio, and return the resized image object.

    Parameters:
    - img: PIL.Image.Image, the image object to resize.
    - new_height: int, the new height of the image.

    Returns:
    - resized_img: PIL.Image.Image, the resized image object.
    """
    # Ensure that the input is a PIL image object
    if not isinstance(img, Image.Image):
        raise ValueError("Input must be a PIL image object")

    # Calculate the new width maintaining the aspect ratio
    original_width, original_height = img.size
    aspect_ratio = original_width / original_height
    new_width = int(new_height * aspect_ratio)

    # Resize the image
    resized_img = img.resize((new_width, new_height))

    return resized_img

def transform_image(image_file):
    # Define the transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to the input size expected by your model
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize the image
                             std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_file)
    image_transformed = transform(image)
    
    # Keep only the first 3 channels
    if image_transformed.shape[0] > 3:
        image_transformed = image_transformed[:3, :, :]
    
    image_batch = image_transformed.unsqueeze(0)  # Add a batch dimension
    return image_batch

def load_retrain_model():
    class ReTrainModel(nn.Module):
        def __init__(self):
            super(ReTrainModel, self).__init__()

            # Load pre-trained pretrained model
            self.pretrain_net = models.alexnet()
            
            # Freeze the pretrained model parameters
            for param in self.pretrain_net.parameters():
                param.requires_grad = True
                
            # Add custom fully connected layer on top
            self.classifier = nn.Sequential(
                nn.Linear(in_features=1000, out_features=600),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(in_features=600, out_features=120),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(in_features=120, out_features=3) # Softmax is applied by the crossentropy loss function, so is not needed here.
            )
            
        def forward(self, x):
            features = self.pretrain_net(x) # Run it through the pretrain
            out = self.classifier(features) # Then the custom classifier
            return out
    
    model = ReTrainModel()

    model_checkpoint = torch.load(r'.\stilted-totem-49_20240306_081830.pth', map_location=torch.device('cpu'))
    model.load_state_dict(model_checkpoint)
    
    return model

def predict_from_model(model, image):
    logits = model(image)
    probabilities = F.softmax(logits, dim=1)
    
    # Convert tensor to a simple list
    probabilities_list = probabilities.squeeze().tolist()

    # Reformat the list to percentages
    probabilities_percentage = [f"{p * 100:.2f}%" for p in probabilities_list]
    
    class_dict = {0:'Glory-Shot', 1:'Butt-Shot', 2:'Topo'}
    result_dict = {class_dict[i]: probabilities_percentage[i] for i in range(len(class_dict))}
    return result_dict, probabilities_list