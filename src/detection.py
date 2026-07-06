"""
Detection module: train and run the YOLOv8 detector for player/goalkeeper/referee/ball.
"""
from ultralytics import YOLO


def train_detector(data_yaml_path, epochs=50, imgsz=640, batch=16, device=0,
                    project='runs', name='football_detector_v1'):
    """Fine-tune YOLOv8n on the football detection dataset."""
    model = YOLO('yolov8n.pt')
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=15,
        device=device,
        project=project,
        name=name,
        exist_ok=True,
        plots=True
    )
    return model, results


def load_detector(weights_path):
    """Load a previously trained detector."""
    return YOLO(weights_path)


def run_inference(model, source, conf=0.35, save=False, **kwargs):
    """Run detection on an image, video, or clip."""
    return model.predict(source=source, conf=conf, save=save, **kwargs)


def run_tracking(model, source, conf=0.35, tracker='bytetrack.yaml', persist=True, **kwargs):
    """Run detection + multi-object tracking (persistent IDs) on a video."""
    return model.track(source=source, conf=conf, tracker=tracker, persist=persist,
                        verbose=False, **kwargs)
