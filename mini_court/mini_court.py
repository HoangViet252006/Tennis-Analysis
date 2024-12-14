import cv2
import sys
import numpy as np

sys.path.append('../')
from utils import get_foot_position, get_center_box, pixels_to_meter
import constants


class MiniCourt():
    def __init__(self, width, height, keypoints_predict):
        self.width = width
        self.height = height

        ratio = (constants.DOUBLE_LINE_WIDTH / constants.COURT_HEIGHT)
        self.courtHeight = int(self.height * 0.6)
        self.courtWidth = int(self.courtHeight * ratio)
        self.yOffset = int((self.height - self.courtHeight) / 2)
        self.xOffset = int((self.width - self.courtWidth) / 2)

        self.courtTL = [self.xOffset, self.yOffset]
        self.courtTR = [self.courtWidth + self.xOffset, self.yOffset]
        self.courtBL = [self.xOffset, self.courtHeight + self.yOffset]
        self.courtBR = [self.courtWidth + self.xOffset, self.courtHeight + self.yOffset]
        self.keypoints_predict = keypoints_predict

    def courtMap(self, image):
        # Define source 4 corners points from keypoints
        pts1 = np.float32([
            [self.keypoints_predict[0 * 2], self.keypoints_predict[0 * 2 + 1]],
            [self.keypoints_predict[1 * 2], self.keypoints_predict[1 * 2 + 1]],
            [self.keypoints_predict[2 * 2], self.keypoints_predict[2 * 2 + 1]],
            [self.keypoints_predict[3 * 2], self.keypoints_predict[3 * 2 + 1]],
        ])
        pts2 = np.float32([self.courtTL, self.courtTR, self.courtBL, self.courtBR])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(image, M, (self.width, self.height))

        return dst, M
    def showLines(self, frame):
        # Draw background
        cv2.rectangle(frame, (0, 0), (self.width, self.height), (255, 255, 255), 6)
        # Draw court
        cv2.rectangle(frame, (self.xOffset, self.yOffset), (self.courtBR[0], self.courtBR[1]), (255, 255, 255), 2)
        # Draw net
        cv2.line(frame, (self.xOffset, self.yOffset + int(self.courtHeight * 0.5)),
                  (self.courtBR[0], self.yOffset + int(self.courtHeight * 0.5)), (255, 255, 255), 2)

        # Draw single court
        self.double_alley_x = self.courtWidth * constants.DOUBLES_ALLEY / constants.DOUBLE_LINE_WIDTH
        cv2.rectangle(frame, (self.xOffset + int(self.double_alley_x), self.yOffset),
                  (self.courtBR[0] - int(self.double_alley_x), self.courtBR[1]), (255, 255, 255), 2)

        # Draw service court
        self.sevice_line_y = self.courtHeight * constants.NO_MANS_LAND_HEIGHT / constants.COURT_HEIGHT
        cv2.rectangle(frame, (self.xOffset + int(self.double_alley_x), self.yOffset + int(self.sevice_line_y)), (
        self.courtBR[0] - int(self.double_alley_x), self.courtBR[1] - int(self.sevice_line_y)),
                  (255, 255, 255), 2)

        # Draw center service line
        cv2.line(frame, (self.xOffset + int(self.courtWidth * 0.5), self.yOffset + int(self.sevice_line_y)),
                  ( self.courtBR[0] - int(self.courtWidth * 0.5), self.courtBR[1] - int(self.sevice_line_y)),
                  (255, 255, 255), 2)
        return frame

    def showPoint(self, frame, point, color):
        cv2.circle(frame, (int(point[0]), int(point[1])), radius=0, color=color, thickness=5)
        return frame

    def givePoint(self, point, frame):
        points = np.float32([[point]])
        dst, M = self.courtMap(frame)
        transformed = cv2.perspectiveTransform(points, M)[0][0]
        return [int(transformed[0]), int(transformed[1])]

    def player_positions_minimap(self, player_detections, frames):
        player_positions_minimap = []
        for frame, player_dict in zip(frames, player_detections):
            player_minimap_dict = {}
            for track_id, bbox in player_dict.items():
                foot_position = get_foot_position(bbox)
                player_minimap_dict[track_id] = self.givePoint([foot_position[0], foot_position[1]], frame)

            player_positions_minimap.append(player_minimap_dict)
        return player_positions_minimap

    def ball_positions_minimap(self, ball_detections, frames):
        ball_positions_minimap = []
        for frame, player_dict in zip(frames, ball_detections):
            ball_minimap_dict = {}
            for track_id, bbox in player_dict.items():
                ball_position = get_center_box(bbox)
                ball_minimap_dict[track_id] = self.givePoint([ball_position[0], ball_position[1]], frame)

            ball_positions_minimap.append(ball_minimap_dict)
        return ball_positions_minimap

    def draw_background_minicourt(self, frame):
        dst, M = self.courtMap(frame)
        output_frame = dst
        cv2.rectangle(output_frame, (0, 0), (self.width, self.height), constants.COLOR_COURT, -1)
        output_frame = self.showLines(output_frame)
        output_frames = output_frame
        return output_frames


    def draw_player_miniMap(self, frames, player_positions_minimnap):
        output_frames = []
        for frame, player_dict in zip(frames, player_positions_minimnap):
            image = frame.copy()
            for track_id, foot_position in player_dict.items():
                output_frame = cv2.circle(image, (int(foot_position[0]), int(foot_position[1])),radius=0, color=(0, 0, 255), thickness=5)
            output_frames.append(output_frame)
        return output_frames

    def draw_ball_miniMap(self, frames, ball_positions_minimnap):
        output_frames = []
        for frame, ball_dict in zip(frames, ball_positions_minimnap):
            image = frame.copy()
            for track_id, ball_position in ball_dict.items():
                output_frame = cv2.circle(image, (int(ball_position[0]), int(ball_position[1])), radius=0,
                                          color=(0, 225, 255), thickness=5)
            output_frames.append(output_frame)
        return output_frames

    def get_distance_meter(self, pixels):
        return pixels_to_meter(pixels,
                        constants.COURT_HEIGHT,
                        self.courtHeight
                        )


            