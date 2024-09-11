# [2024 캡스톤]👕옷맞춤👖__ OpenCV
- 프로젝트 소개

      OpenCV, mediapipe를 활용한 이미지 분석을 통한 추천형 온라인 쇼핑몰
      전신 사진을 업로드 하면 회원가입 시 등록한 신장, 허리, 몸무게 정보를 바탕으로 이미지의 픽셀 당 길이를 계산
      분석 결과로부터 신체의 각 부위 위치를 얻은 후 각 부위 위치를 바탕으로 신체 사이즈가 측정
      추천 형식은 오버핏, 정핏 3개씩이며 총 6개의 옷 추천이 가능


## 💻개발 환경 
<img src="https://img.shields.io/badge/Python-3766AB?style=for-the-badge&logo=Python&logoColor=white"/> <img src="https://img.shields.io/badge/opencv-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white"/>

## 기능 설명
#### waistLine.py과 BodyMeasurementsFromHeight.py 파일 참고.
- 입력받은 사진과 키 정보를 바탕으로 신체치수 (허리 둘레, 팔 길이, 다리 길이) 측정
- 입력받은 사진에서 Mediapipe로 처리된 어깨와 엉덩이 영역을 기반으로 허리 위치 찾기
## 👫영상처리 팀
- 송성민
- 유지원
- 김아연
