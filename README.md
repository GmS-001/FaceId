# Facial Recognition App

This project implements a facial recognition system using TensorFlow and OpenCV.

## Setup Instructions

1. Create a virtual environment:
```bash
python -m venv venv
```

2. Activate the virtual environment:
- On macOS/Linux:
```bash
source venv/bin/activate
```
- On Windows:
```bash
venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the Jupyter notebook:
```bash
jupyter notebook
```

## Project Structure
- `data/`: Directory containing training data
  - `positive/`: Positive samples
  - `negative/`: Negative samples
  - `anchor/`: Anchor samples
- `Facial Recognition App.ipynb`: Main Jupyter notebook 