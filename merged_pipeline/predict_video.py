import warnings
import json
from pathlib import Path

import numpy as np
import hydra
import torch
import cv2
from omegaconf import DictConfig

from feature_descriptors.feature_descriptors import HogFeatureDescriptor


def predict_class(img, feature_descriptor, model, class_map):
    features = feature_descriptor.predict(img)[np.newaxis, ...].astype(np.float32)
    pred = model.predict(features)[1][0][0]

    return class_map[pred]


@hydra.main(version_base=None, config_path='.', config_name='config.yaml')
def main(config: DictConfig):

    # Set up the object detector model
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=UserWarning)
        object_detector = torch.hub.load(
            'ultralytics/yolov5', 'custom', path=config.models.yolov5.weights_path, force_reload=True
        )
    object_detector.to(config.device)

    hog = HogFeatureDescriptor()
    svm = cv2.ml.SVM_load(config.models.svm.weights_path)
    with open(config.models.svm.class_map_path, 'r') as f:
        class_map = json.load(f)
    class_map = {int(k): v for k, v in class_map.items()}

    test_video_path = Path(config.test_video_path)
    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    test_video_cap = cv2.VideoCapture(str(test_video_path))
    result_cap = cv2.VideoWriter(
        str(save_dir / f"{test_video_path.stem}.mp4"),
        cv2.VideoWriter_fourcc(*'mp4v'),
        50,
        (int(test_video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(test_video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    )
    while True:
        success, img = test_video_cap.read()
        if success:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            break

        detections = json.loads(object_detector(img).pandas().xyxy[0].to_json(orient='records'))
        for det in detections:
            x_min, y_min, x_max, y_max = int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax'])

            # Crop the image
            crop = img[y_min: y_max, x_min: x_max]
            class_name = predict_class(crop, hog, svm, class_map)

            img = cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
            text_width, text_height = cv2.getTextSize(
                class_name, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=1)[0]
            y_center = (y_min + y_max) // 2
            text_origin = (x_min + 17, y_center + 17)
            box_coords = (
                (text_origin[0], text_origin[1]),
                (text_origin[0] + text_width + 2, text_origin[1] - text_height - 2)
            )
            background_color = (0, 0, 0)
            img = cv2.rectangle(img, box_coords[0], box_coords[1], background_color, cv2.FILLED)
            color = (255 - background_color[0], 255 - background_color[1], 255 - background_color[2])
            cv2.putText(
                img, class_name, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, tuple(color), 1, cv2.LINE_AA
            )
            result_cap.write(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    result_cap.release()


if __name__ == '__main__':
    main()
