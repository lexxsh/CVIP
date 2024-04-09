## Alexnet 코드 실습

### Alexnet 논문으로 바탕으로 한 CIFAR-10 데이터셋을 활용하여 성능을 향상시키는 실습을 진행하였습니다.

여러가지 파인튜닝이 허용되있으며 본 실습결과로 중간 미팅 발표자를 선정할 계획입니다... 


| 날짜        | 활동                                | 결과                                           | 피드백                                           |
|-------------|-------------------------------------|-------------------------------------------------|-------------------------------------------------|
| 2024-04-05  | Alexnet model을 바탕으로한 실습 코드 작성 + 데이터셋 적용 |**Train Loss**: 0.2104 **Train Accuracy**: 92.812% <br>   **Test Loss**: 0.5733 **Test Accuracy**: 83.42% |
| 2024-04-06  | 다양한 Augmentation 조정 : 성능감소 <br> batch-size 조정: 변화 없음 |**Train Loss**: 0.4849 **Train Accuracy**: 83.224 % <br> **Test Loss**: 0.6027 **Test Accuracy**: 79.51 %|
| 2024-04-07  | 모델 파인튜닝#1 -> 자제한 내용은 아래에 정리 예정. |**Train Loss**:  0.1734 **Train Accuracy**: 94.204 % <br> **Test Loss**: 0.5207 **Test Accuracy**: 84.38 %|
| 2024-04-08  | 모델 파인튜닝#2 -> 자제한 내용은 아래에 정리 예정. |Epoch 8에서 터짐.. **Train Loss**:  0.6761 **Train Accuracy**: 77.06 % <br> **Test Loss**: 0.6053 **Test Accuracy**: 79.4 % |


Model Fine Tuning #1

| 레이어 | 첫 번째 모델 | 두 번째 모델 |
|-------|-------------------|-------------------|
| Conv1 | Conv2d(11x11), ReLU, LocalResponseNorm, MaxPool2d(3x3) | Conv2d(3x3), ReLU, Conv2d(3x3), ReLU, MaxPool2d(3x3) |
| Conv2 | Conv2d(5x5), ReLU, LocalResponseNorm, MaxPool2d(3x3) | Conv2d(3x3), ReLU, Conv2d(3x3), ReLU, MaxPool2d(2x2) |
| Conv3 | Conv2d(3x3), ReLU | Conv2d(3x3), ReLU, Conv2d(3x3), ReLU, MaxPool2d(2x2) |
| Conv4 | Conv2d(3x3), ReLU | Conv2d(3x3), ReLU, Conv2d(3x3), ReLU, MaxPool2d(2x2) |
| Conv5 | Conv2d(3x3), ReLU, MaxPool2d(3x3) | Conv2d(3x3), ReLU, Conv2d(3x3), ReLU, MaxPool2d(2x2) |

| FC1   | Linear, ReLU, Dropout | Linear, ReLU, Dropout |
| FC2   | Linear, ReLU, Dropout | Linear, ReLU, Dropout |
| FC3   | Linear | Linear |

아래에 파라미터의 총 개수를 비교하는 표를 작성하겠습니다. 파라미터의 개수는 각 레이어의 가중치및 편향에 따라 결정됩니다.

| 모델        | 파라미터 수 |
|-------------|------------|
| 첫 번째 모델 | 약 58.1M   |
| 두 번째 모델 | 약 15.9M   |

Model Fine Tuning #2

| 레이어 | 첫 번째 모델 | 두 번째 모델 |
|-------|-------------------|-------------------|
| Conv1 | Conv2d(3x3), Batch nomalization, ReLU, Conv2d(3x3), Batch nomalization, ReLU, MaxPool2d(3x3), |
| Conv2 | Conv2d(3x3), Batch nomalization, ReLU, Conv2d(3x3), Batch nomalization, ReLU, MaxPool2d(3x3), |
| Conv3 | Conv2d(3x3), Batch nomalization, ReLU, Conv2d(3x3), Batch nomalization, ReLU, MaxPool2d(3x3),  |
| Conv4 | Conv2d(3x3), Batch nomalization, ReLU, Conv2d(3x3), Batch nomalization, ReLU, Conv2d(3x3), Batch nomalization, ReLU, MaxPool2d(3x3) |
| Conv5 | Conv2d(3x3), Batch nomalization, ReLU, Conv2d(3x3), Batch nomalization, ReLU, Conv2d(3x3), Batch nomalization, ReLU, MaxPool2d(3x3) |

| FC1   | Linear, ReLU, Dropout | Linear, ReLU, Dropout |
| FC2   | Linear, ReLU, Dropout | Linear, ReLU, Dropout |
| FC3   | Linear | Linear |

colab이 epoch 8에서 붕괴되버렸습니다...
