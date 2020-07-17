# [프로젝트] 닮은 개 종류 추천 사이트 Doggie Face

- 팀원: 강규현, 김광현, 김희준, 정찬오, 조민재 

  <br>

## 1. 소개



<img src="https://user-images.githubusercontent.com/58925328/87747557-a45f9600-c82e-11ea-9f9c-55f4a637fa58.PNG" style="zoom: 67%;" />

- 사용기술: CNN(딥러닝), Python, OpenCV, Django

- 서비스: 닮은 개 종류 추천 

- URL: http://heejun1942.pythonanywhere.com/ 

- 설명: CNN 모델로 개 종류별 얼굴을 학습하여, 사람얼굴과 유사한 개 종류를 알려줍니다. 

<br><br>

## 2. 코드 설명

>- Google Custom Search API 을 사용하여 개 이미지 수집
>- dlib와 OpenCV를 이용하여 개 얼굴과 사람 얼굴 인식 
>
>- OpenCV를이용하여 밝기와 블러 처리 변화로 학습 이미지 데이터 증식

<br>

### 모델 학습

>CNN모델을 사용하여, 강아지 종류별로 생김새를 학습한다.



(1) `img_dl_gcs.py`: Google Custom Search API 을 사용하여 개 이미지 수집

(2) `img_face_detector.py`: dlib로 개 얼굴을 인식하여, 학습에 사용하기 위해 개 얼굴을 64x64 사이즈로 자름. 

(3) `03_img_resize.py`: 자른 이미지가 64x64보다 작을 경우가 존재할 수 있으므로 다시 한번 크기를 64x64로 조정해줌.

(4) `img_generator.py`: OpenCV를 이용하여 밝기와 블러 처리 변화로 학습 이미지 데이터를 증식시킴. (과적합 방지)

(5) `img_model_generator.py`: CNN 모델 학습

(6) `img_face_judgement.py`: 사람 이미지를 넣어 모델에서 강아지 종류를 예측함.

<br>

### 웹 어플리케이션 구현

>웹 어플리케이션 구현을 위해 Django를 사용함



- `django`폴더에 장고 웹어플리케이션을 위한 코드 존재.
- pythonanywhere를 통해 배포함.

