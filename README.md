# 🎨 Neural Style Transfer - CodTech Internship Task 3

This project implements a **Neural Style Transfer (NST)** model using PyTorch, allowing you to apply the artistic style of one image to the content of another. Built as part of **CodTech Internship Task 3**, the project demonstrates the use of convolutional neural networks (CNNs) for artistic image transformation.

---

## 🧠 What is Neural Style Transfer?

Neural Style Transfer (NST) is a technique that blends two images:
- **Content Image**: The base image (like a photo of a person or scene)
- **Style Image**: The artistic reference (like Van Gogh’s Starry Night)
- Output: A new image that preserves the **content** of the first and the **style** of the second.

This is done using a pretrained **VGG-19** model to extract and combine the image features.

---

## 📁 Project Structure
<pre>
├── style_transfer.py       # Main Python script  
├── content.jpg             # Your content image  
├── style.jpg               # Your style image  
├── output.jpg              # The generated stylized image  
└── README.md               # Project documentation  
</pre>

---

## ⚙️ How It Works

1. Load and preprocess the content and style images
2. Extract feature maps using a pretrained **VGG-19** network
3. Compute:
   - **Content Loss**: Difference in feature maps between content and target image
   - **Style Loss**: Difference in Gram matrices (texture) between style and target
4. Use optimization to update the target image
5. Save and display the final image

---

## 🚀 How to Run

### 1. Requirements

Make sure to install the following:
```bash
pip install torch torchvision matplotlib pillow
```
### 2. Place Your Images

Put your images in the same folder as style_transfer.py, and name them:  
•content.jpg (photo you want to stylize)  
•style.jpg (art/image you want the style from)  

### 3. Run the Script
```bash
python3 style_transfer.py
```
The output will be: 
•Saved as: output.jpg  
•Also displayed automatically  

---

### 📚 Technologies Used
•Python  
•PyTorch  
•TorchVision (VGG19)  
•PIL & Matplotlib

---

### 👨‍💻 Author

Raman Kumar
CodTech Internship - AI Track
July 2025
