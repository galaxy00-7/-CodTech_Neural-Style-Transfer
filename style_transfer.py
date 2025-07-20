import torch
from PIL import Image
from torchvision import transforms

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and transform an image
def load_image(path, max_size=400, shape=None):
    image = Image.open(path).convert("RGB")
    
    # Resize image
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)

    if shape is not None:
        size = shape  # already a (H, W) tuple
    else:
        size = (size, size)  # convert scalar to tuple

    # Transform: resize -> to tensor -> normalize
    in_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Apply transform and add batch dimension
    image = in_transform(image).unsqueeze(0)
    return image.to(device)

from torchvision import models

# Load pretrained VGG19 model
vgg = models.vgg19(pretrained=True).features.to(device).eval()

# Freeze parameters (we don’t want to train them)
for param in vgg.parameters():
    param.requires_grad = False

# Define which layers we want to extract
# Keys are layer indices in VGG19
content_layer = '21'  # conv4_2
style_layers = {
    '0': 'conv1_1',
    '5': 'conv2_1',
    '10': 'conv3_1',
    '19': 'conv4_1',
    '28': 'conv5_1'
}
# Extract features from specific layers
def get_features(image, model, layers):
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

# Calculate Gram Matrix
def gram_matrix(tensor):
    # Get batch_size, depth, height, width
    _, d, h, w = tensor.size()

    # Reshape so that each row is a channel
    tensor = tensor.view(d, h * w)

    # Compute the Gram matrix
    gram = torch.mm(tensor, tensor.t())

    return gram

# Load your content and style images
content = load_image("content.jpg")
style = load_image("style.jpg", shape=(content.size(2), content.size(3)))

# Extract content and style features
content_features = get_features(content, vgg, {content_layer: 'content'})
style_features = get_features(style, vgg, style_layers)

# Compute style Gram matrices
style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

# Create target image as a clone of content
target = content.clone().requires_grad_(True).to(device)

import torch.optim as optim

# Define style weights (importance of each style layer)
style_weights = {
    'conv1_1': 1.0,
    'conv2_1': 0.75,
    'conv3_1': 0.2,
    'conv4_1': 0.2,
    'conv5_1': 0.2
}

# Content and style weights
content_weight = 1e4
style_weight = 1e2

# Define optimizer (we optimize the target image!)
optimizer = optim.Adam([target], lr=0.003)

import matplotlib.pyplot as plt

# Optimization loop
steps = 300  # You can change this to 500+ for better results

for step in range(1, steps + 1):
    # Get features of the target image
    target_features = get_features(target, vgg, {**style_layers, content_layer: 'content'})

    # Content loss (from conv4_2)
    content_loss = torch.mean((target_features['content'] - content_features['content']) ** 2)

    # Style loss
    style_loss = 0
    for layer in style_layers:
        conv_name = style_layers[layer]  # e.g., 'conv1_1'
        target_feature = target_features[conv_name]
        target_gram = gram_matrix(target_feature)
        style_gram = style_grams[conv_name]
        layer_loss = torch.mean((target_gram - style_gram) ** 2)
        style_loss += style_weights[conv_name] * layer_loss

    # Total loss
    total_loss = content_weight * content_loss + style_weight * style_loss

    # Backpropagation
    optimizer.zero_grad()
    total_loss.backward(retain_graph=True)
    optimizer.step()

    # Print progress
    if step % 50 == 0:
        print(f"Step {step}/{steps}, Total loss: {total_loss.item():.4f}")

    # Convert target tensor to a displayable image
def im_convert(tensor):
    image = tensor.cpu().clone().detach()
    image = image.squeeze(0)  # remove batch dimension

    # Undo normalization
    unloader = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )
    image = unloader(image)
    image = torch.clamp(image, 0, 1)  # restrict values between 0 and 1

    return transforms.ToPILImage()(image)

# Save and show final result
final_img = im_convert(target)
final_img.save("output.jpg")
final_img.show()
