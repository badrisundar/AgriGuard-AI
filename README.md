# AgriGuard AI
## Intelligent Smart Agriculture Platform

AgriGuard AI is an AI-powered plant disease detection platform
that analyzes uploaded leaf images using deep learning.

### Technology Stack

- Frontend: React 19, Vite
- Backend: Python, FastAPI
- Database: PostgreSQL
- Deep Learning: PyTorch, MobileNetV3
- Image Processing: Pillow

### Main Features

- Plant leaf validation
- 39-class plant disease classification
- Healthy leaf identification
- Confidence-based uncertain predictions
- Disease descriptions, symptoms and prevention
- Management guideline retrieval
- Prediction history

### Requirements

- Python 3.11
- Node.js and npm
- PostgreSQL
- Required model checkpoints

### Backend Setup

Create and activate a Python virtual environment:

    python3.11 -m venv .venv311
    source .venv311/bin/activate

Install dependencies:

    pip install -r requirements.txt

Start PostgreSQL and create the database named agriguard.

Initialize and seed the database if it has not already been populated.

Start the backend from the project root:

    uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000

Backend documentation:

    http://127.0.0.1:8000/docs

### Frontend Setup

Open another terminal:

    cd frontend
    npm install
    npm run dev

Open:

    http://localhost:5173

### Required Model Checkpoints

Place these files in the models directory:

- mobilenetv3_full39_best.pth
- leaf_validator_v3_best.pth

### Important Note

AgriGuard AI provides decision-support predictions.
It is not a replacement for professional agricultural advice.
Predictions should be independently verified, especially
before making crop-management decisions.
