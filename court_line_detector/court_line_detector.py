import cv2
import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import Compose, ToTensor, Normalize, Resize


class Court_line():
    def __init__(self, model_path):
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, 14 * 2)
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint["model_params"])
        self.transfrom = Compose([
            ToTensor(),
            Resize((224, 224), antialias=True),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = self.transfrom(image_rgb).unsqueeze(0)
        with torch.no_grad():
            output = self.model(image_tensor)
        keypoints = output.squeeze().cpu().numpy()
        original_h, original_w = image.shape[:2]
        keypoints[::2] *= original_w / 224.0
        keypoints[1::2] *= original_h / 224.0
        for i in range(0, len(keypoints), 2):
            x, y = int(keypoints[i]), int(keypoints[i + 1])
        return keypoints

    def draw_keypoints(self, image, keypoints):
        for i in range(0, len(keypoints), 2):
            x, y = int(keypoints[i]), int(keypoints[i + 1])
            cv2.putText(image, str(i // 2), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        return image

    def draw_keypoints_onVideo(self, frames, keypoints):
        output_video = []
        for frame in frames:
            output_video.append(self.draw_keypoints(frame, keypoints))
        return output_video
