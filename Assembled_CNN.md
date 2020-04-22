## Assembled CNN

CNN과 관련된 여러 기법을 하나의 네트워크로 조립하는 좀 더 광범위하고 체계적인 연구를 수행했다. 도입된 많은 기법을 고려할 때, 우리는 먼저 기법을 네트워크 트윗과 정규화라는 두 가지 범주로 나누었다.
 
<p align="center"><img src="https://user-images.githubusercontent.com/45933225/79968789-94ecdd00-84cb-11ea-9138-d2b190762176.png" width="50%"></p>

정규화는 AutoAugment [4], Mixup [39] 등의 데이터 증강 프로세스를 통해 교육 데이터를 증가시키거나, 드롭아웃 [31], DropBlock [5] 등의 프로세스로 CNN의 복잡성을 제한함으로써 오버피팅을 방지하는 방법이며, 광범위한 실험을 통해 이 두 종류의 기법을 조립하는 과정을 체계적으로 분석하고, 우리의 접근법이 상당한 성능 향상으로 이어진다는 것을 증명한다. FLOPS가 실제 GPU 장치의 추론 속도에 비례하지 않는다고 보았기 때문에, 일반적으로 사용되는 FLOPS 측정(초당 부동 소수점 작업) 대신 처리량(이미지/초)을 사용했다.

    - 모델
    공식 TensorFlow[1] ResNet 3을 베이스 코드로 사용한다

    - 데이터 셋
    ImageNet ILSVRC-2012[29] 데이터 세트를 사용하며, 1.3M 교육 이미지와 1,000개의 클래스를 가지고 있다
    전처리 훈련 단계에서 직사각형 영역은 무작위로 3/4에서 4/3까지의 가로 세로 비율을 사용하여 무작위로 잘라지며, 전체 이미지에 걸쳐 잘라낸 영    역의 비율은 5%에서 100%로 무작위로 선택, 그런 다음 절단 부위는 무작위 확률 0.5로 수평으로 플립되는 224x224 사각형 영상으로 크기를 조정    한다. 
    유효성 검사 중에는 먼저 가로 세로 비율이 유지되는 동안 각 이미지의 크기를 256픽셀로 조정하고 이미지를 224x224 크기로 자르고 RGB 채널을     훈련과 동일하게 정상화한다.

    - 배치사이즈, 에폭 수, 모멘텀
    하이퍼 파라미터 교육을 위해 1,024개의 배치 크기를 사용한다
    기본 훈련 에폭 수는 120
    momentum 0.9의 stochastic gradient descent 사용된다

    - 모델의 성능을 측정하는 데 사용되는 측정지표
    Top-1 : ImageNet ILSVRC-2012 [29] 검증 데이터 세트의 분류 정확도를 측정하는 것이다
    처리량 : GPU 장치에서 초당 처리되는 이미지 수로 정의된다
    mCE : 손상된 이미지에 대한 분류 모델의 성능을 측정한다

### Network Tweaks

- ResNet-D

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/79968845-a504bc80-84cb-11ea-8e65-bd4371ab1d84.png" width="50%"></p>

1. 처음 두 개의 convolution stride 사이즈가 바뀌었다. (그림 2(b)의 파란색) 
2. 2×2 Average pooling layer는 Convolution(녹색) 전에 2의 stride로 추가되었다. 
3. 7x7의 큰 Convolution은 stem layer(빨간색)에서 3개의 작은 3x3 Convolution으로 대체되었다.

#### Channel Attention (SE, SK)
1. SE(Squeeze and Excitation) 네트워크는 채널과 같은 관계를 모델링하여 네트워크의 표현 능력을 높이는 데 초점을 맞춘다. SE는 채널 정보만을 얻기 위해 글로벌 풀링을 통해 공간 정보를 제거한 후, 이 모듈에서 완전히 연결된 두 레이어가 채널 간의 상관 관계를 학습한다. 
2. Selective kernel(SK)은 인간의 시각 피질에서 뉴런의 수용 크기가 서로 다르다는 사실에서 영감을 얻는다. SK 유닛은 커널 크기가 다른 여러 개의 지점이 있으며, 소프트맥스 주의를 이용해 모든 지점이 융합되어 있다

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/79968865-ac2bca80-84cb-11ea-8d9e-371e991321d2.png" width="70%"></p>
 
<p align="center"><img src="https://user-images.githubusercontent.com/45933225/79968932-c5347b80-84cb-11ea-8183-96a045e180cf.png" width="50%"></p>

표 2는 channel attention의 다른 구성에 대한 결과를 보여준다. SE는 SK에 비해 처리량은 높지만 정확도는 낮다(표 2의 C1). Top-1의 정확도와 처리량 사이의 관계를 고려하여 R50+SK†이 사용된다.

#### Anti-Alias Downsampling (AA)
1. AA는 심층 네트워크의 이동-균형도를 개선할 것을 제안한다
2. AA는 저역-통과 필터로서 그들 사이에 기존의 strided-Conv와 함께 실질적인 앨리어스 방지 효과를 얻기 위해 제안된다.

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/79968958-cd8cb680-84cb-11ea-8f40-a9e2a8de91e3.png" width="50%"></p>

#### Big Little Network (BL)
1. BigLittleNet[3]은 컴퓨터 비용을 절감하고 정확도를 높이는 것을 목표로 하는 동시에 해상도가 다른 여러 지점을 적용한다. 

### Regularization

- AutoAugment (Autoaug)
1. AutoAugment[4]는 데이터로부터 증강 전략을 학습하는 데이터 증강 절차다
- Mixup
1.데이터 확대를 위한 교육 세트의 두 가지 예를 보간하여 하나의 예를 만든다
2. 훈련 데이터 세트의 특징 공간에 빈 공간을 채우는 신경 네트워크 보간 예제를 보여줌으로써 이러한 문제를 완화시킨다
3. 첫 번째 유형은 두 개의 미니 배치를 사용하여 혼합된 미니 배치를 생성한다
4. 두 번째 유형의 구현에서는 CPU 리소스를 덜 사용한다. 왜냐하면 하나의 혼합된 미니 배치를 생성하려면 하나의 미니 배치만 미리 처리하면 되기 때문이다
- DropBlock
1. DropBlock [7]은 연속적인 활성화 영역을 삭제하여 특정 의미 정보를 제거할 수 있어 네트워크의 정규화에 효율적이다
- Label Smoothing (LS)
1. 무한 출력을 억제하고 오버 피팅을 방지한다
- Knowledge Distillation (KD)
1. 한 신경망(Teacher)에서 다른 신경망(Student)으로 지식을 전달하는 기술이다. 
2. Teacher model은 복잡하지만 정확도가 높은 번거로운 모델이며, 약하지만 가벼운 Student model은 교사 모델을 모방하여 자신의 정확도를 높일 수 있다.


### Experiment Results

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/79968991-d9787880-84cb-11ea-9356-0b7c20a21411.png" width="50%"></p>

ResNet-D와 SK를 스택으로 하면 ResNet-D와 SK를 별도로 적용하여 얻은 성능 이득의 합계와 거의 동등한 수준으로 정확도 상위 1위 이득을 증가시킨다. 그 결과, 두 개의 트윗은 서로 거의 영향을미치지 않고 독립적으로 성능을 향상시킬 수 있다는 것을 보여준다.

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/79969017-e1d0b380-84cb-11ea-9a63-9e7675cfbb95.png" width="100%"></p>

최종 모델은 표 7에 E11로 나와 있으며, 이 모델을 Asemble-ResNet-50이라고 부른다.

### Conclusion
CNN을 위한 다양한 기법을 단일의 couvolutional 네트워크에 조립하는 것이 ImageNet ILSVRC2012 검증 데이터 집합에서 Top-1 정확도와 mCE의 개선을 이끈다는 것을 보여준다. 시너지 효과는 단일 네트워크에서 다양한 네트워크 트윗과 정규화 기법을 함께 사용함으로써 달성되었다. 
