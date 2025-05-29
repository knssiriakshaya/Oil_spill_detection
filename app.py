# app.py
from flask import Flask, render_template, request
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import snntorch as snn
from snntorch import surrogate

# === Flask App ===
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# === Classes ===
classes = ['No Oil Spill', 'Oil Spill']  # Adjust if more classes
num_classes = len(classes)

# === Define SNN Model (same as your training code) ===
class SNN(nn.Module):
    def __init__(self):
        super().__init__()
        beta = 0.95
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.AdaptiveAvgPool2d((4, 4))
        self.lif3 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

        dummy = torch.zeros(1, 1, 64, 64)
        x = self.pool1(self.bn1(self.conv1(dummy)))
        x, _ = self.lif1(x)
        x = self.pool2(self.bn2(self.conv2(x)))
        x, _ = self.lif2(x)
        x = self.pool3(self.bn3(self.conv3(x)))
        x, _ = self.lif3(x)
        flat_size = x.view(1, -1).shape[1]

        self.fc1 = nn.Linear(flat_size, num_classes)
        self.lif4 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

    def forward(self, x, num_steps=15):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        spk4_sum = 0

        for _ in range(num_steps):
            x1 = self.pool1(self.bn1(self.conv1(x)))
            spk1, mem1 = self.lif1(x1, mem1)

            x2 = self.pool2(self.bn2(self.conv2(spk1)))
            spk2, mem2 = self.lif2(x2, mem2)

            x3 = self.pool3(self.bn3(self.conv3(spk2)))
            spk3, mem3 = self.lif3(x3, mem3)

            flat = spk3.view(spk3.size(0), -1)
            x4 = self.fc1(flat)
            spk4, mem4 = self.lif4(x4, mem4)

            spk4_sum += spk4

        return spk4_sum / num_steps

# === Load Trained Model ===
model = SNN()
model.load_state_dict(torch.load("best_snn_model (1).pth", map_location='cpu'))
model.eval()

# === Image Transform ===
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# === Routes ===
@app.route('/')
def home():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return 'No image uploaded', 400

    image_file = request.files['image']
    img_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
    image_file.save(img_path)

    # Preprocess and predict
    image = Image.open(img_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        _, pred = torch.max(output, 1)
        predicted_class = classes[pred.item()]

    return render_template('result.html', prediction=predicted_class, image_file=img_path)

if __name__ == '__main__':
    app.run(debug=True)
