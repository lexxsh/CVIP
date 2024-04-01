## CONTENT

---

## Abstarct

본 글에서 요약할 논문은 "ImageNet Classification with Deep Convolutional Neural Networks"입니다.

이 논문은 Alexnet 모델에 대해 다루었으며 이 모델을 사용하여 2010년  ILSVRC 대회에서는 Top 1, Top 5 test error가 각각 37.5 %, 17.0%를 기록하였으며 2012년 ILSVRC 대회에서는 Top 5 test error(모델이 예측한 5개의 클래스 범위 내에서 정답이 없는 경우의 오류율) 기준으로 15.4%로 1위를 차지하였습니다.

5개의 합성곱 계층, max-pooling, softmax 등을 사용하였으며 Conv 연산에 효율적인 GPU를 구현해냈습니다.

또한 Overfitting을 방지하기 위하여 dropout이라는 새로운 정규화 방식을 도입했다고 합니다.

CNN 구조와 자세한 모델의 내용에 대해 살펴봅시다.

---

# Key Points

### Dataset

22,000개 이상의 카테고리에 속하는 1,500만 장의 레이블이 지정된 고해상도 이미지 데이터셋인 ImageNet 📓

### Network

## 1. Introduction

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/15f285db-8602-4adb-ba6a-10d845474bea/acdbef81-51b6-466e-9ebc-89edb41e1d70/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/15f285db-8602-4adb-ba6a-10d845474bea/97adf7ed-52bf-47c1-807c-af9c92e6d94b/Untitled.png)

- Standard feedforward neural network (FNN) -  다층 퍼셉트론
    
    입력층에서 출력층으로 신호가 한 방향으로만 흐르는 인공신경망
    
    각 층의 뉴런이 이전 층의 모든 뉴런과 연결되어 있어 정보를 처리하고 패턴을 학습할 수 있는 강력한 모델 
    
    주로 분류(classification)와 회귀(regression) 문제를 해결하는 데 사용
    
- Convolutional neural network (CNN)
    
    이미지 인식 및 분류에 사용되는 딥러닝 모델
    
    합성곱층(convolutional layers), 풀링층(pooling layers), 그리고 완전 연결층(fully connected layers)으로 구성
    
    합성곱층은 입력 이미지의 특징을 추출하고, 풀링층은 공간적 차원을 줄이고 계산 비용을 줄임
    완전 연결층은 추출된 특징을 기반으로 입력 이미지를 분류하거나 예측 
    

최근까지 레이블이 지정된 이미지 데이터셋은 비교적 작은 편이였습니다. (CLFAR-10, NORB 등등) 간단한 인식작업은 이러한 데이터셋으로도 잘 수행이 되었습니다. 그러나 현실 상황에서 객체는 다양한 변이를 보입니다. 따라서 이에 맞게 수많은 데이터셋이 존재해야합니다. 이때 LabelMe를 포함한 ImageNet 데이터셋이 나오게 됩니다. 이를 학습하기 위해 더 용량이 큰 모델이 필요하게 됩니다. 깊이와 너비를 조절하며 성능을 끌어올리기 시작했습니다.

이 논문에서 중요하게 다룬 점은 다음과 같습니다.

1. ImageNet의 하위집합에 대해 가장 큰 합성곱 신경망 중 하나를 훈련
2. 2D 합성곱, 신경망 훈련에 내재된 GPU 구현
3. Overfitting을 해결하기 위한 다양한 방법
4. 5개의 Conv Layer, 3개의 fc layer의 깊이 중요성 및 layer 제거 시 성능 저하

fid foward network → 1차원의 인풋이 들어가는 반면에

CNN → 2차원의 인풋이 들어간다.

1. **구조**: FNN은 각 층의 뉴런이 이전 층의 모든 뉴런과 연결된 완전 연결 구조를 가지고 있습니다. 반면에 CNN은 입력 데이터의 공간적 구조를 고려하여 합성곱층과 풀링층을 사용합니다.
2. **데이터 처리**: FNN은 입력 데이터의 형태를 고려하지 않고, 각 뉴런 간의 연결로만 데이터를 처리합니다. 그러나 CNN은 합성곱층을 사용하여 입력 데이터의 지역적 패턴 및 공간적 관계를 보존하고 인식합니다.
3. **파라미터 공유**: CNN은 합성곱층에서 사용되는 필터를 통해 파라미터 공유를 이용하여 입력 이미지의 특징을 추출합니다. 이는 학습 가능한 파라미터의 수를 줄이고 모델의 일반화 성능을 향상시킵니다. 반면 FNN은 각 뉴런 간의 연결로 모든 파라미터가 고유합니다.
4. **이미지 처리**: CNN은 주로 이미지 처리 및 컴퓨터 비전 작업에 특화되어 있습니다. 반면에 FNN은 이미지 외의 다양한 종류의 데이터에도 적용할 수 있습니다.

## 2. The Dataset

22,000개의 카테고리에 속하는 1500만 개 이상의 레이블이 지정된 ImageNet 데이터셋 사용

이미지분류 성능을 평가하기 위한 top-1, top-5 error 사용

top-1 error - 새로운 테스트 이미지들에 예측한 1개의 클래스가 실제 클래스와 동일한지 판단하는 오류율

top-5 error - 분류기가 높은 확률로 예측한 상위 5개의 클래스중 실제 클래스가 존재하는지 판단하는 오류율

데이터셋의 이미지 크기는 각각 다름, 이미지를 256 X 256 고정 해상도로 다운 샘플링

일정한 이미지가 들어오면 가로나, 세로의 크기가 256이 되도록 조정 후 CentorCrop하여 입력으로 사용

이외에 다른 전처리 방식은 사용하지 않았다.

## 3. The Architecture

각 network에 사용된 방법을 차례로 소개합니다.

### 3-1 ReLU Nonlinearity

standard 방법은 tanh 혹은 sigmoid를 사용하는 방법입니다. 그러나 이 활성함수들은  gradient descent로 학습을 진행할때 속도가 매우 느려집니다.

따라서 본 논문에서는 이를 해결하기 위해 non-saturating 이며 비선형성인 ReLU를 참조합니다.  

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/15f285db-8602-4adb-ba6a-10d845474bea/907db592-f8c4-4b37-8e7f-7b0a2b797fd8/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/15f285db-8602-4adb-ba6a-10d845474bea/5bf43c5c-848b-4397-a0d4-25419c7e761d/Untitled.png)

오른쪽 그래프를 보게되면 ReLU(solid line)이 25% 오류에 도달하기 까지의 속도가 tanh(dashed line)보다 매우 빠른것을 확인할 수 있습니다.

타 논문인 Jarrett에서는 tanh가 특정 데이터셋에서 잘 작동한다고 주장하였으나 본 논문에선 Overfitting 방지 및 학습속도 가속화에 집중하였으므로 케이스가 좀 다르다고 합니다.

*** 참고할 내용***

여러가지 활성함수가 존재하는데 왜 ReLU인지?

vanishing gradient problem - 시그모이드나 쌍곡선 탄젠트 함수는 입력이 크거나 작을 때 그래디언트가 매우 작아질 수 있는데 입력이 양수인 경우에는 그래디언트가 1로 유지 해 이를 해결 

오버피팅을 확실히 줄이고 비선형성 유지 및 효율적인 학습이 가능하다.

### 3-2 Training on Multiple GPUs

하나의 network를 두개의 GPU에 분산.

GPU 여러 개를 사용한 병렬처리 기법으로 학습시간을 획기적으로 줄였다.

### 3-3 Local Response Normalization

참고만 하고 이해만 하고 넘어가도됨→ 최근 단계에선 Batch nomalization을 주로 사용한다. https://taeguu.tistory.com/29

자세한 건 위 페이지 참고 

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/15f285db-8602-4adb-ba6a-10d845474bea/c12adba2-03be-490d-a85b-1805e069da02/Untitled.png)

측면 억제 (Lateral Inhibition)는 주변 픽셀 간의 경쟁적인 상호 억제를 의미 → LRN에서 중요한 포인트
LRN은 사용하는 이유는 이미지의 인접화소들을 억제시키고 특징을 부각시키기 위함입니다. 그 결과, 정확도가 1.4% 향상되었습니다. LRN은 신경생물학에서 그 원리를 가져왔습니다. 예컨대, 밝은 빛을 보면 눈이 어두워진다거나, 특정 사물에 집중하면 그 부분만 집중하여 보이게 되는 현상

헤르만 격자를 통해서 참고하면 될 듯합니다.

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/15f285db-8602-4adb-ba6a-10d845474bea/3b579956-1d99-4876-a02d-d77afb7c38f2/Untitled.png)

### 3-4 Overlapping Pooling

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/15f285db-8602-4adb-ba6a-10d845474bea/d997cc48-1ead-49cd-8792-a1ebe751d3c3/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/15f285db-8602-4adb-ba6a-10d845474bea/7c6f528d-5b45-45bc-bfd5-cb04a6fd7bda/Untitled.png)

Pooling Layer - 동일한 커널 맵 내의 이웃하는 뉴런 그룹의 출력을 요약 및 Summary 제공

보통 Pooling Layer 는 stride를 조절하여 overlap을 하지 않으나  본 눈문에서는 z=3, s=2로 stride를 줄여 overlapping pooling을 구성하여 오버피팅을 줄일 수 있도록 하였다. 이로 인한 정확도는 0.4% 향상하는 효과를 거둘 수 있었다.

### 3-5 Overall Architecture

전체적인 구조는 

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/15f285db-8602-4adb-ba6a-10d845474bea/441577e6-c4a8-4e59-8798-cfb9a7a400f5/Untitled.png)

다음과 같으며 자세한 내용은 다음 포스팅에 코드와 함께 작성하도록 하겠습니다.

### 4-1 Data Augmentation

오버피팅을 줄이기 위한 가장 쉬운 방법은 데이터의 수를 증가 시키는 것입니다.

논문을 통해 확인해보면 두가지 방법으로 이를 해결하였다고 작성되어 있습니다.

1. Image Augmentation
    
    2장에서 만들어진 256x256 이미지를 224x224 크기로 RandomResizedCrop하고 RandomHorizontalFlip을 적용시킨 것입니다. 그렇게 되면 하나의 이미지에서 2048장의 이미지를 얻을 수 있게 됩니다. 모델을 테스트할 때는 좌측 상단, 좌측 하단, 우측 상단, 우측 하단, 중앙의 5가지 위치에서 224x224 크기의 이미지를 얻고, RandomHorizontalFlip을 적용시켜 총 10가지의 이미지를 얻고 각각의 이미지로부터 얻은 softmax 값의 평균으로 최종 label을 결정합니다.
    동일한 이미지들을 조금씩 변형시켜가며 학습하면 Overfitting을 방지하는 데 도움이 됩니다. Data Augmentation에는 이미지를 좌우 반전시키는 Mirroring 기법, 이미지의 특정 부분을 무작위로 자르는 Random Crops 기법을 사용하였습니다.
    
2. RGB 채널을 변경하는 PCA 기법
    
    선형대수의 주성분분석에서 나온 방법으로 https://ddongwon.tistory.com/114
    이 블로그에 잘 작성되어 있으므로 참고하면 되겠습니다.
    
    이미지에서 RGB 채널의 intensity를 변경 후 RGB값에 PCA를 적용하였습니다.
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/15f285db-8602-4adb-ba6a-10d845474bea/0166740c-87a2-43c7-9501-ec8dc0597d92/Untitled.png)
    
    위 값들을 각각 이미지 픽셀해 저장하였습니다.
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/15f285db-8602-4adb-ba6a-10d845474bea/1b7013c4-4211-4330-b56d-0db6ecdd1ca5/Untitled.png)
    

### 4-2 Dropout

dropout을 간단하게 소개하자면 서로 연결된 layer들 중에서 0~1 사이의 확률로 hidden layer의 뉴런에서 발생하는 출력을 제거하는 것입니다. 

본 논문에서는 이 확률을 0.5로 지정하였고  이를 통하여 역전파가 참여하지않아 Co-adaptation을 줄 일수 있게 됩니다. 따라서 확실한 특징만을 학습하여 오버피팅(과학습)을 줄 일 수 있었습니다.

alextnet에서는 첫 두 fc layer 뒤에 dropout을 적용하였습니다!

dropout에서 자세한 내용은 https://heytech.tistory.com/127 을 참고하면 좋을 것 같습니다.
