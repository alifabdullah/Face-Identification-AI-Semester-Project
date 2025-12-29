🎭 Actor Image Classifier Web Application

A full-stack AI web application that identifies celebrities from images using a trained deep learning model. The backend is powered by FastAPI + PyTorch, and the frontend is built with HTML & CSS.

📂 Dataset Used

🗂 Celebrity Face Image Dataset (Kaggle)
🔗 https://www.kaggle.com/datasets/vishesh1412/celebrity-face-image-dataset

Contains labeled celebrity face images

Images organized by actor name

Used for training and validation

🧠 Trained Model

📦 Model Type: Vision Transformer (ViT-B/16)
💾 Model Format: Pickle / PyTorch model file

🔗 Download Model File:
👉 https://drive.google.com/file/d/1MFt1NwGKz98OyCgYIG0icIANA0C3tsu9/view?usp=sharing

⚠️ Important:
After downloading, update the model path inside app.py before running the server.

⚙️ Technologies Used

🐍 Python

🚀 FastAPI

🔥 PyTorch

🧠 Vision Transformer (ViT)

🌐 HTML, CSS

🔄 Application Workflow

1️⃣ User uploads an image from the frontend
2️⃣ Image is sent to FastAPI backend
3️⃣ Backend preprocesses the image (resize + normalize)
4️⃣ Trained model predicts the actor
5️⃣ Result (name + confidence) is returned and displayed

▶️ How to Run the Project
1️⃣ Install Dependencies
pip install fastapi uvicorn torch torchvision pillow

2️⃣ Update Model Path

Edit app.py and set the correct path to the downloaded model file.

3️⃣ Run the Server
python app.py


🌐 The browser will open automatically at:
http://127.0.0.1:8000

🔌 API Endpoint

📌 POST /predict

Input: Image file

Output:

Actor name

Confidence score

⚠️ Notes

🎓 Academic / learning project

📉 Limited dataset size

🧪 Performance depends on image quality and lighting

❌ Not production-ready
