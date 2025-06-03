# 이미지 분류 모델을 통한 딥러닝 실습 프로젝트


## :flags:프로젝트 개요
![image](https://github.com/user-attachments/assets/7ca044e3-d9f7-4834-8398-d28a361ebe1d) 출처: https://www.datamaker.io/blog/posts/17

*  딥러닝을 활용한 이미지 분류 기술을 학습하고 실제로 구현하는 것을 목적으로 합니다. 특히 이미지 내 동물 객체를 분류하는 작업을 통해 CNN의 구조와 작동 원리를 이해하고, 이를 바탕으로 성능 개선을 위한 다양한 전처리 및 데이터 증강 기법을 적용할 예정입니다


## :mag:해당 프로젝트 선정 이유

![image](https://github.com/user-attachments/assets/83ebfdf7-a854-4ef3-900f-d700f9858066) 출처: 슈퍼브 블로그


* 이미지 분류는 의료 진단, 자율주행차 객체 인식, CCTV 보안 감시 등 다양한 산업 분야에서 필수적인 기술로, 특히 CNN 기반의 딥러닝 기술이 활발히 활용되고 있습니다

* 이미지 분류 학습을 위해 자주 활용되는 개 vs 고양이 프로젝트를 활용할 것이며, 단순 구현에 그치지 않고 전처리 및 데이터 증강 과정을 통해 성능을 향상시키는 방향으로 발전시켜보고자 합니다

* 실제 산업 프로젝트에서 자주 활용되는 전처리 전략과 모델 최적화 과정의 실전 경험을 얻고자 이번 프로젝트를 선택했습니다

## :chart_with_upwards_trend:데이터셋 

* 출처 : [kaggle Dogs vs Cats 이미지 분류 데이터셋](https://www.kaggle.com/c/dogs-vs-cats/data)
* 총 데이터 수: 훈련 데이터셋 25000장(크기:545MB), 테스트 데이터셋 12500장(크기:272MB)

## :books:모델 설명 및 구현 계획

:green_book: 모델 구조
* 기본 모델은 CNN 기술 모델을 활용할 것이며, 출력은 개 vs 고양이 두 가지 클래스로 분류하는 이진 분류로 출력
* Convolution Layer + Polling Layer 이용한 CNN 모델
  
:closed_book: 성능 향상을 위한 전처리 기법
* 이미지 정규화 : 데이터를 정규분포처럼 변환하여 데이터 학습 속도와 안정성을 높임
* 이미지 리사이징 : 모든 이미지 데이터들을 250x250 크기로 고정된 크기로 재조정하여 오류를 줄이고 더욱 정확한 비교가 가능하도록 만듬

:blue_book: 기존 Kaggle 프로젝트 대비 차별점
* 이미지 정규화 및 회전 같은 전처리는 그대로 사용, 하지만 추가적으로 이미지 리사이징을 통해 추가적으로 성능을 향상
* VCG16방식의 사전에 학습된 이미지 분류 모델 사용이 아닌, Convolution Layer + Polling Layer 이용한 기초적인 CNN 구조 이용

:orange_book: 모델 구현 계획
* 데이터 구현 및 전처리
  + Kaggle에서 제공하는 "Dogs vs Cats" 이미지 데이터셋 활용
  + 이미지 크기 250x250으로 통일하여 모델 입력 정규화
  + RGB 이미지 픽셀값 정규화 (0~1 범위)
* 모델 구성
  + Sequential 모델로 구성
  + Convolution Layer + Polling Layer 기법을 모델 조합에 이용
  + 출력층은 이진 분류에 적합한 Sigmoid 함수를 사용하여 (0:고양이 1:개) 출력
  + Dropout 기법을 이용하여 과적합 방지. 위 코드의 Flatten() 아래줄에 Dropout(0.5) 수식을 추가
    + 노드를 줄여 과적합 방지 뿐만 아니라, 모델이 특정 패턴에 의존하는 것을 줄일 수 있다
  
  ```python
    model = Sequential
    ([
      Conv2D(32, (3,3), activation='relu', input_shape=(250,250,3)),
      MaxPooling2D(2,2),
      Conv2D(64, (3,3), activation='relu'),
      MaxPooling2D(2,2),
      Conv2D(128, (3,3), activation='relu'),
      MaxPooling2D(2,2),
      Flatten(),
      Dense(512, activation='relu'),
      Dense(1, activation='sigmoid')
    ])

 ## :books: 모델 구현 결과 및 비교 분석

:green_book: 기본 CNN 모델 성능
* 구성: Conv2D → MaxPooling → Conv2D → MaxPooling → Conv2D → MaxPooling → Flatten → Dense → Dropout → Dense

  + 정규화: 픽셀값을 0~1 범위로 정규화

  + 리사이징: 모든 이미지를 250x250으로 고정

  + Dropout: 0.5 적용하여 과적합 방지

* 결과 요약

  + 최고 Validation Accuracy: 약 XX%

  + 과적합 방지에 효과적이나, 데이터 양이 한정적일 경우 높은 일반화 성능은 어려움

:closed_book: VGG16 전이 학습 모델 성능
* 구성: VGG16 base (include_top=False) + GlobalAveragePooling2D + Dropout + Dense

  + 사전 학습된 가중치: ImageNet 사용

  + fine-tuning 없이 feature extractor로만 사용

* 결과 요약

  + 최고 Validation Accuracy: 약 YY%

  + 사전 학습된 모델 덕분에 적은 에포크 수로도 높은 성능 확보 가능

  + 기본 CNN 모델에 비해 더 안정적이고 높은 정확도를 보임

## :arrow_upper_right: 기대 효과
이번 프로젝트를 통해 이미지 분류에 필수적인 CNN(Convolutional Neural Network) 구조에 대한 실질적인 이해를 바탕으로, 딥러닝 기반의 이미지 분류 모델을 처음부터 끝까지 직접 구현하는 경험을 얻게 됩니다. 단순히 모델을 구성하고 학습하는 것을 넘어, 전처리 및 데이터 증강을 통한 성능 개선, Dropout과 같은 과적합 방지 기법 적용, 그리고 전이 학습 모델(VGG16)과의 비교 분석을 통해 모델 선택 및 최적화의 실전 감각을 익힐 수 있습니다.

또한, Kaggle 데이터셋을 기반으로 실무에서 자주 접할 수 있는 데이터 정제, 폴더 구조 재정리, 평가용 데이터 분할 등의 실제 프로젝트에서 요구되는 데이터 핸들링 역량도 함께 강화할 수 있습니다.

최종적으로는 CNN을 이용한 이미지 분류 모델의 구축부터 평가까지의 전 과정을 경험함으로써, 향후 의료 영상 분석, 보안 모니터링, 자율주행 등 다양한 산업 분야에 적용 가능한 컴퓨터 비전 기술에 대한 기반을 탄탄히 다질 수 있을 것입니다.

