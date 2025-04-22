from ultralytics import YOLO

# Load a model
model = YOLO('models/yolov8n-seg.pt')  # load a pretrained model (recommended for training)

if __name__ == '__main__':
    # Train the model
    results = model.train(data='configs/bread-seg.yaml', epochs=100, imgsz=640, batch=16)