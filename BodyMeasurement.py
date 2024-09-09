import cv2
import mediapipe as mp
import numpy as np
import json
import sys

# MediaPipe 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)
mp_drawing = mp.solutions.drawing_utils

def calculate_distance(p1, p2):
    """두 점 사이의 유클리디언 거리를 계산"""
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def estimate_body_measurements(image, user_height_cm):
    """이미지를 기반으로 신체 치수 추정"""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # 어깨 너비 추정
        shoulder_left = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
        shoulder_right = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
        shoulder_width_pixels = calculate_distance(shoulder_left, shoulder_right)

        # 허리 둘레 추정 (간단한 예시로 양쪽 엉덩이의 거리를 사용)
        hip_left = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y]
        hip_right = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y]
        waist_circumference_pixels = calculate_distance(hip_left , hip_right) # 가정: 허리는 엉덩이의 두 배로 계산

        # 팔 길이 추정
        wrist_left = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y]
        shoulder_left = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
        arm_length_pixels = calculate_distance(wrist_left, shoulder_left)

        # 다리 길이 추정
        hip_left = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y]
        ankle_left = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y]
        leg_length_pixels = calculate_distance(hip_left, ankle_left)

        top_length_left = calculate_top_length(left_shoulder[0], left_shoulder[1], left_hip[0], left_hip[1])
        top_length_right = calculate_top_length(right_shoulder[0], right_shoulder[1], right_hip[0], right_hip[1])
        average_top_length = (top_length_left + top_length_right) / 2

        # 픽셀 단위의 치수를 사용자의 키를 기반으로 실제 치수로 변환
        heel_left = [landmarks[mp_pose.PoseLandmark.LEFT_HEEL].x, landmarks[mp_pose.PoseLandmark.LEFT_HEEL].y]
        height_in_pixels = calculate_distance([landmarks[mp_pose.PoseLandmark.NOSE].x, landmarks[mp_pose.PoseLandmark.NOSE].y], heel_left)
        pixel_to_cm_ratio = (user_height_cm - 13) / height_in_pixels

        shoulder_width_cm = shoulder_width_pixels * pixel_to_cm_ratio
        waist_circumference_cm = waist_circumference_pixels * pixel_to_cm_ratio + 12
        arm_length_cm = arm_length_pixels * pixel_to_cm_ratio
        leg_length_cm = leg_length_pixels * pixel_to_cm_ratio

        # JSON 데이터 생성
        body_measurements = {
            "shoulderWidth": shoulder_width_cm,
            "waistWidth": waist_circumference_cm,
            "sleeveLength": arm_length_cm,
            "bottomLength": leg_length_cm,
            "topLength": average_top_length
            # "chestCrossSection": 추가 예정
        }

        return body_measurements
    else:
        return None

def main():
    # 이미지 파일 경로
    image_path = r'C:\image\women1.jpg'
    # 이미지 읽기
    image = cv2.imread(image_path)

    if image is not None:
        # 사용자의 키 입력 받기 (단위: cm)
        user_height_cm = float(input("사용자의 키를 입력하세요 (단위: cm): "))

        body_measurements = estimate_body_measurements(image, user_height_cm)
        if body_measurements:
            # JSON 형식으로 변환
            body_measurements_json = json.dumps(body_measurements, ensure_ascii=False)
            print(body_measurements_json)
        else:
            print("신체 치수를 추정할 수 없습니다.")
    else:
        print("이미지를 읽을 수 없습니다.")

if __name__ == "__main__":
    main()
