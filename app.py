import webbrowser
import threading
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import vit_b_16
from PIL import Image
import io
import uvicorn

# -------------------------------
# FastAPI app
# -------------------------------
app = FastAPI()

# Serve the templates folder at /static
app.mount("/static", StaticFiles(directory=r'C:\Users\User\Downloads\Python\Python\Python\templates', html=True), name="static")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Redirect root / to HTML page


@app.get("/")
async def root():
    return RedirectResponse("/static/index.html")

# -------------------------------
# Model setup
# -------------------------------
model = vit_b_16(weights="IMAGENET1K_V1")
model.heads.head = nn.Linear(768, 17)

state_dict = torch.load(
    r"C:\Users\User\Downloads\Python\Python\Python\Backend\enhanced_vit_model.pth",
    map_location=torch.device('cpu')
)
model.load_state_dict(state_dict)
model.to("cpu")
model.eval()

classes = [
    'Angelina Jolie',
    'Brad Pitt',
    'Denzel Washington',
    'Hugh Jackman',
    'Jennifer Lawrence',
    'Johnny Depp',
    'Kate Winslet',
    'Leonardo DiCaprio',
    'Megan Fox',
    'Natalie Portman',
    'Nicole Kidman',
    'Robert Downey Jr.',
    'Sandra Bullock',
    'Scarlett Johansson',
    'Tom Cruise',
    'Tom Hanks',
    'Will Smith'
]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# -------------------------------
# Prediction endpoint
# -------------------------------


@app.post("/predict")
async def predict(file: UploadFile):
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    x = transform(img).unsqueeze(0).to("cpu")

    with torch.no_grad():
        out = model(x)

    probs = torch.softmax(out, dim=1)
    confidence, idx = torch.max(probs, dim=1)

    return {"name": classes[idx.item()], "confidence": float(confidence.item())}

# -------------------------------
# Open browser automatically
# -------------------------------


def open_browser():
    webbrowser.open("http://127.0.0.1:8000/")


# -------------------------------
# Start Uvicorn server
# -------------------------------
if __name__ == "__main__":
    threading.Timer(1.5, open_browser).start()
    uvicorn.run(app, host="127.0.0.1", port=8000)
