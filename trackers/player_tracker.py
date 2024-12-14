import cv2
from ultralytics import YOLO
import pickle
from utils import get_center_box, cal_distance
class PlayerTracker():
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def filter_player(self, court_detections, player_detections):
        player_detections_first_frame = player_detections[0]
        chosen_player = self.choosen_player(court_detections, player_detections_first_frame)
        filtered_player_detections = []
        for player_dict in player_detections:
            filtered_player_dict = {track_id:bbox for track_id, bbox in player_dict.items() if track_id in chosen_player}
            filtered_player_detections.append(filtered_player_dict)
        return filtered_player_detections

    def choosen_player(self, court_detections, player_dict):
        distances = []
        for track_id, bbox in player_dict.items():
            center_coordinate_player = get_center_box(bbox)
            min_distance = float('inf')
            for i in range(0, len(court_detections), 2):
                keypoint = (court_detections[i], court_detections[i + 1])
                distance = cal_distance(center_coordinate_player, keypoint)
                if min_distance > distance:
                    min_distance = distance
            distances.append((track_id, min_distance))

        distances.sort(key = lambda x : x[1])

        chosen_player = (distances[0][0], distances[1][0])
        return chosen_player


    def detection_frames(self, frames, is_stub = False, stubs_path = None):
        player_detections = []
        if is_stub and stubs_path is not None:
            with open(stubs_path, 'rb') as f:
                player_detections = pickle.load(f)
                return player_detections

        for frame in frames:
            player_detections.append(self.detection_frame(frame))

        if stubs_path is not None:
            with open(stubs_path, 'wb') as f:
                pickle.dump(player_detections, f)
        return player_detections

    def detection_frame(self, frame):
        results = self.model.track(frame, persist=True)[0]
        class_name = results.names
        player_dict = {}

        for box in results.boxes:
            result = box.xyxy.tolist()[0]
            track_id = int(box.id.tolist()[0])
            cls_id = int(box.cls.tolist()[0])
            cls_name = class_name[cls_id]

            if cls_name == 'person':
                player_dict[track_id] = result
        return player_dict

    def drawing_boundingbox(self, frames, player_detections):
        output_frames = []
        for frame, player_dict in zip(frames, player_detections):
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = map(int, bbox)
                cv2.putText(frame, f"Player ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            output_frames.append(frame)
        return output_frames


