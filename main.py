import argparse
from mini_court import MiniCourt
from utils import (read_video,
                   save_video,
                   cal_distance,
                   draw_player_stats,
                   )
import constants
from trackers import BallTracker, PlayerTracker
from copy import deepcopy
from court_line_detector import Court_line
import pandas as pd
import cv2

def get_args():
    parser = argparse.ArgumentParser(description="Tennis detection")
    parser.add_argument("--input_video_path", "-i", type=str, required=True)
    parser.add_argument("--output_video_path", "-o", type=str, required=True)
    args = parser.parse_args()
    return args

def main(args):
    video_frames = read_video(args.input_video_path)

    # detections
    players = PlayerTracker('weights/yolov8x.pt')
    player_detections = players.detection_frames(video_frames, False, 'tracker_stubs/player_detections.pkl')
    balls = BallTracker('weights/tennis_weights.pt')
    ball_detections = balls.detection_frames(video_frames, False, 'tracker_stubs/ball_detections.pkl')
    ball_detections = balls.interpolate(ball_detections)
    keypoints = Court_line('weights/court_line_weights.pt')
    keypoints_predict = keypoints.predict(video_frames[0])
    # choose player
    player_detections = players.filter_player(keypoints_predict, player_detections)

    # MiniCourt
    minicourt = MiniCourt(250, 500, keypoints_predict)
    back_ground_minicourt = minicourt.draw_background_minicourt(video_frames[0])
    player_positions_minimap = minicourt.player_positions_minimap(player_detections, video_frames)
    ball_positions_minimap = minicourt.ball_positions_minimap(ball_detections, video_frames)

    # draw minicourt live
    minicourt_live = back_ground_minicourt.copy()
    minicourt_live = [minicourt_live] * len(video_frames)
    minicourt_live = minicourt.draw_player_miniMap(minicourt_live, player_positions_minimap)
    minicourt_live = minicourt.draw_ball_miniMap(minicourt_live, ball_positions_minimap)

    # frame hit ball
    frames_hit_ball = balls.get_ball_shot_frames(ball_detections)
    player_stats_data = [{
        'frame_num':0,
        'player_1_last_shot_speed':0,
        'player_1_speed':0,
        'player_1_distance': 0,

        'player_2_last_shot_speed':0,
        'player_2_speed':0,
        'player_2_distance': 0
    }]

    for ball_shot_idx in range(len(frames_hit_ball) - 1):
        start_frame = frames_hit_ball[ball_shot_idx]
        end_frame = frames_hit_ball[ball_shot_idx + 1]
        ball_shot_time_in_seconds = (end_frame - start_frame) / constants.FPS

        # Get distance covered by the ball
        distance_covered_by_ball_pixels = cal_distance(ball_positions_minimap[start_frame][1],
                                                           ball_positions_minimap[end_frame][1])
        distance_covered_by_ball_meters = minicourt.get_distance_meter(distance_covered_by_ball_pixels)

        # Speed of the ball shot in km/h
        speed_of_ball_shot = distance_covered_by_ball_meters / ball_shot_time_in_seconds * 3.6
        # player who the ball
        player_positions = player_positions_minimap[start_frame]
        player_shot_ball = min(player_positions.keys(),
                               key=lambda player_id: cal_distance(player_positions[player_id],
                                                                      ball_positions_minimap[start_frame][1]))

        # opponent player speed
        opponent_player_id = 1 if player_shot_ball == 2 else 2
        distance_covered_by_opponent_pixels = cal_distance(
            player_positions_minimap[start_frame][opponent_player_id],
            player_positions_minimap[end_frame][opponent_player_id])
        distance_covered_by_opponent_meters = minicourt.get_distance_meter(distance_covered_by_opponent_pixels)


        speed_of_opponent = distance_covered_by_opponent_meters / ball_shot_time_in_seconds * 3.6

        current_player_stats = deepcopy(player_stats_data[-1])
        current_player_stats['frame_num'] = start_frame
        current_player_stats[f'player_{player_shot_ball}_last_shot_speed'] = speed_of_ball_shot
        player_stats_data.append(current_player_stats)


    player_stats_data_df = pd.DataFrame(player_stats_data)
    frames_df = pd.DataFrame({'frame_num': list(range(len(video_frames)))})
    player_stats_data_df = pd.merge(frames_df, player_stats_data_df, on='frame_num', how='left')
    player_stats_data_df = player_stats_data_df.ffill()
    player_stats_data_df['player_1_distance'] = 0.0
    player_stats_data_df['player_2_distance'] = 0.0
    window_frame = 5

    for player_dict in player_positions_minimap:
        number_of_frames = len(player_positions_minimap)
        for frame_index in range(0, number_of_frames, window_frame):
            last_frame = min(frame_index + window_frame, number_of_frames - 1)

            for track_id, player_positions in player_dict.items():
                start_position = player_positions_minimap[frame_index][track_id]
                end_position = player_positions_minimap[last_frame][track_id]
                distance_pixels = cal_distance(start_position, end_position)
                distance_meter = minicourt.get_distance_meter(distance_pixels)
                time_move = (last_frame-frame_index) / constants.FPS
                speed_player = distance_meter / time_move * 3.6
                speed_column = f"player_{track_id}_speed"
                distance_column = f"player_{track_id}_distance"
                if frame_index > 0:
                    # distance
                    prev_frame = int(frame_index) - 1
                    while prev_frame > 0 and player_stats_data_df.loc[prev_frame, distance_column] == 0:
                        prev_frame -= 1
                    prev_value = player_stats_data_df.loc[prev_frame, distance_column]
                    player_stats_data_df.loc[frame_index, distance_column] = distance_meter + prev_value
                    # speed
                    player_stats_data_df.loc[frame_index, speed_column] = speed_player



    player_stats_data_df['player_1_distance'] = player_stats_data_df['player_1_distance'].replace(0, None)
    player_stats_data_df.loc[0, 'player_1_distance'] = 0.0
    player_stats_data_df['player_2_distance'] = player_stats_data_df['player_2_distance'].replace(0, None)
    player_stats_data_df.loc[0, 'player_2_distance'] = 0.0
    player_stats_data_df['player_1_speed'] = player_stats_data_df['player_1_speed'].replace(0, None)
    player_stats_data_df.loc[0, 'player_1_speed'] = 0.0
    player_stats_data_df['player_2_speed'] = player_stats_data_df['player_2_speed'].replace(0, None)
    player_stats_data_df.loc[0, 'player_2_speed'] = 0.0
    player_stats_data_df = player_stats_data_df.ffill()

    # player_stats_data_df.to_csv('analysis/player.csv', sep='\t', encoding='utf-8', index=False, header=True)

    # draw bbox players
    output_frames = players.drawing_boundingbox(video_frames, player_detections)
    output_frames = balls.drawing_boundingbox(output_frames, ball_detections)
    output_frames = keypoints.draw_keypoints_onVideo(output_frames, keypoints_predict)

    # Draw Player Stats
    output_frames = draw_player_stats(output_frames, player_stats_data_df)

    # Draw Minimap on top right corner
    for miniMap, frame in zip(minicourt_live, output_frames):
        x_offset = frame.shape[1] - miniMap.shape[1]
        y_offset = 0
        frame[y_offset:miniMap.shape[0], x_offset:frame.shape[1]] = miniMap

    # Draw frame number on top left corner
    for i, frame in enumerate(output_frames):
        cv2.putText(frame, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # save video
    save_video(args.output_video_path, output_frames)


if __name__ == '__main__':
    args = get_args()
    main(args)