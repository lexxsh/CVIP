## Alexnet 코드 실습

### Alexnet 논문으로 바탕으로 한 CIFAR-10 데이터셋을 활용하여 성능을 향상시키는 실습을 진행하였습니다.

여러가지 파인튜닝이 허용되있으며 본 실습결과로 중간 미팅 발표자를 선정할 계획입니다... 


| 날짜        | 활동                                | 결과                                           | 피드백                                           |
|-------------|-------------------------------------|-------------------------------------------------|-------------------------------------------------|
| 2024-04-05  | Alexnet model을 바탕으로한 실습 코드 작성 + 데이터셋 적용 |**Train Loss**: 0.2104 **Train Accuracy**: 92.812% <br>   **Test Loss**: 0.5733 **Test Accuracy**: 83.42% |
| 2024-04-06  | 다양한 Augmentation 조정 : 성능감소 <br> batch-size 조정: 변화 없음 |**Train Loss**: 0.4849 **Train Accuracy**: 83.224 % <br> **Test Loss**: 0.6027 **Test Accuracy**: 79.51 %|
| 2024-04-07  | 모델 파인튜닝 -> 자제한 내용은 아래에 정리 예정. |**Train Loss**:  0.1734 **Train Accuracy**: 94.204 % <br> **Test Loss**: 0.5207 **Test Accuracy**: 84.38 %|


Model Fine Tuning #1

| Layer | Operation (Net) | Operation (AlexNet) | Output Shape (Net) | Output Shape (AlexNet) | Parameters (Net) | Parameters (AlexNet) |
|-------|-----------------|---------------------|--------------------|------------------------|-------------------|-----------------------|
| Layer 1 | Conv2d(64 channels, 3x3 kernel, stride 1, padding 1) ReLU | Conv2d(96 channels, 11x11 kernel, stride 4) ReLU | (64, H/2, W/2) | (96, H/4, W/4) | 1792 | 34944 |
| Layer 2 | Conv2d(128 channels, 3x3 kernel, padding 1) ReLU | Conv2d(256 channels, 5x5 kernel, stride 1, padding 2) ReLU | (128, H/4, W/4) | (256, H/8, W/8) | 73856 | 153856 |
| Layer 3 | Conv2d(128 channels, 3x3 kernel, padding 1) ReLU | Conv2d(384 channels, 3x3 kernel, padding 1) ReLU | (128, H/8, W/8) | (384, H/8, W/8) | 147584 | 307328 |
| Layer 4 | Conv2d(256 channels, 3x3 kernel, padding 1) ReLU | Conv2d(384 channels, 3x3 kernel, padding 1) ReLU | (256, H/8, W/8) | (384, H/8, W/8) | 295168 | 614656 |
| Layer 5 | Conv2d(512 channels, 3x3 kernel, padding 1) ReLU | Conv2d(256 channels, 3x3 kernel, padding 1) ReLU | (512, H/8, W/8) | (256, H/8, W/8) | 1180160 | 1226752 |
| AdaptiveAvgPool2d | AdaptiveAvgPool2d(6x6) | AdaptiveAvgPool2d(6x6) | (512, 6, 6) | (256, 6, 6) | 0 | 0 |
| Fully Connected Layers | Linear(6x6x512 -> 4096) ReLU Dropout | Linear(6x6x256 -> 4096) ReLU Dropout | (4096,) | (4096,) | 25169920 | 37752832 |
| | Linear(4096 -> 4096) ReLU Dropout | Linear(4096 -> 4096) ReLU Dropout | (4096,) | (4096,) | 16781312 | 16781312 |
| | Linear(4096 -> num_classes) | Linear(4096 -> num_classes) | (num_classes,) | (num_classes,) | 40970 | 40970 |

In this table, each operation is explicitly mentioned for both the modified architecture (Net) and the original AlexNet.
