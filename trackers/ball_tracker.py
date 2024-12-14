import cv2
from ultralytics import YOLO
import pickle
import pandas as pd

class BallTracker():
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def interpolate(self, ball_detections):
        ball_position = [x.get(1, []) for x in ball_detections]

        df_ball_position = pd.DataFrame(ball_position, columns=['x1', 'y1', 'x2', 'y2'])

        df_ball_position = df_ball_position.interpolate()
        df_ball_position = df_ball_position.bfill()

        ball_position = [{1: x} for x in df_ball_position.to_numpy().tolist()]

        return ball_position

    def get_ball_shot_frames(self, ball_position):
        ball_position = [x.get(1, []) for x in ball_position]

        df_ball_position = pd.DataFrame(ball_position, columns=['x1', 'y1', 'x2', 'y2'])
        df_ball_position['mid_y'] = (df_ball_position['y1'] + df_ball_position['y2']) / 2
        df_ball_position['mid_y_rolling_mean'] = df_ball_position['mid_y'].rolling(window=5, min_periods=1, center=False).mean()
        df_ball_position['deltal_y'] = df_ball_position['mid_y_rolling_mean'].diff()
        df_ball_position['ball_hit'] = 0
        min_chaneg_for_hit = 25

        for i in range(1, len(df_ball_position) - int(min_chaneg_for_hit  * 1.2)):
            negative_position_change = df_ball_position['deltal_y'].iloc[i] < 0 and  df_ball_position['deltal_y'].iloc[i + 1] > 0
            positive_position_change =  df_ball_position['deltal_y'].iloc[i] > 0 and  df_ball_position['deltal_y'].iloc[i + 1] < 0

            if negative_position_change or positive_position_change:
                count_change = 0
                for change_frame in range(i + 1, i + int(min_chaneg_for_hit * 1.2) + 1):
                    negative_change = df_ball_position['deltal_y'].iloc[i] < 0 and  df_ball_position['deltal_y'].iloc[change_frame] > 0
                    positive_change = df_ball_position['deltal_y'].iloc[i] > 0 and \
                                               df_ball_position['deltal_y'].iloc[change_frame] < 0

                    if negative_change and negative_position_change:
                        count_change += 1
                    elif positive_change and positive_position_change:
                        count_change += 1
                if count_change > min_chaneg_for_hit - 1:
                    df_ball_position.loc[i, 'ball_hit'] = 1

        frames_hit_ball = df_ball_position[df_ball_position['ball_hit'] == 1].index.tolist()

        return frames_hit_ball





    def detection_frames(self, frames, is_stub = False, stubs_path = None):
        ball_detections = []
        if is_stub and stubs_path is not None:
            with open(stubs_path, 'rb') as f:
                ball_detections = pickle.load(f)
                return ball_detections

        for frame in frames:
            ball_detections.append(self.detection_frame(frame))

        if stubs_path is not None:
            with open(stubs_path, 'wb') as f:
                pickle.dump(ball_detections, f)
        return ball_detections

    def detection_frame(self, frame):
        results = self.model.predict(frame, conf=0.15)[0]
        ball_dict = {}

        for box in results.boxes:
            result = box.xyxy.tolist()[0]
            ball_dict[1] = result
        return ball_dict

    def drawing_boundingbox(self, frames, player_detections):
        output_frames = []
        for frame, player_dict in zip(frames, player_detections):
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = map(int, bbox)
                cv2.putText(frame, f"Tennis ball", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            output_frames.append(frame)
        return output_frames