# CNN(Convolutional Neural Network)

신경망을 통해 학습을 하게 되면 어려운 많은 문제들을 해결이 가능함.

영상에 기반한 인식 알고리즘에서 좋은 결과를 얻으려면, 사전에 많은 처리 과정이 필요하기 때문에 multi-layered neural network를 바로 적용하는 것은 어려움이 따름.

### Multi-layered Neural Network의 문제점(특히, 비전에 적용에 따른..)
training data만을 넣어주는게 아닌, prior knowledge을 이용해 네트워크의 구조를 특수한 형태로 변형을 시켜주는 것.

그렇다면 2-D 이미지가 갖는 특성을 최대한 이끌어낼 수 있는 방법은?
(2-D의 기본적인 내용은 생략함.)

기존 신경망은 위 그림 처럼, 픽셀 값의 약간의 이동만 있더라도 새로운 training data로 처리를 해줘야 하는 문제점을 가지고 있었음.
또한 입력해주는 데이터가 항상 같은 경우가 아니기 때문에 좋은 결과를 얻기는 더욱 어려웠음.

결과적으로 multi-layered neural network의 문제점은 이미지의 topology(컴퓨터 네트워크의 요소들(링크, 노드 등)을 물리적으로 연결해 놓은 것.)을 고려하지 않고 row data에 많은 training data를 필요로 하고, 학습 시간을 필요로 함.
ex) 32 x 32 image => 2^32*32 무수한 양의 패턴이 나옴. - 그래서 전처리 과정이 따로 필요.

결과적으로 기존의 fully-connected muylti-layered neural network을 사용하면 다음과 같은 3가지 측면에 문제가 발생함.

    - 학습 시간(Training time)
    - 망의 크기(Network size)
    - 변수의 개수(Number of free parameters)
    
##### 위와 같은 문제를 해결하고자 CNN (Convolutional Neural Network) 이다.

CNN 들어가기 전에, Receptive Field(수용 영역)에 대해서 알아야 함.
사전 의미로는 간단하게 말하자면 다음과 같음.

    - 정보처리와 관계되는 포에 대한 응답을 일으키는 자극의 영역.
    
쉽게 말하자면 외부 자극이 전체 영향을 끼치는 것이 아니라 특정 영역에만 영향을 준다는 뜻임.
그래서 이게 '인식 알고리즘'으로 이어짐.

    - 영상 전체 영역에 대해 서로 동일한 연관성(중요도)로 처리하는 대신에 특정 범위에 한정 처리를 한다면 훨씬 좋은 결과를 얻을 수 있다라는 짐작.


Convolution 이란?

'신호 및 시스템'과정에서의 convolution은 특정 시스템에 입력이 가해졌을 때 시스템의 반응이 어떻게 되는지 해석하기 위한 용도로 사용.

'영상 처리 분야'에서 convolution은 주로 filter 연산에 사용이 되며, 영상으로부터 특정 feature들을 추출하기 위한 필터를 구현할 경우 사용.  - 3 x 3 window or mask or kernel을 영상 전체에 대해 반복적으로 수행을 하게 되며 그 계수(weight)값들의 따라 적정한 결과를 얻을 수 있음.

자세한 내용은 생략 하겠음. - 영상 처리 부분, 신호 및 시스템 분야을 공부하자ㅎㅎ

#### CNN의 특징

Locality(Local Connectivity) - 수용 영역과 유사하게 local 정보를 활용. 공간적으로 인접한 신호들에 대한 correlation 관계를 비선형 필터를 적용하여 추출해냄.

Shared Weights - 동일한 계수를 갖는 filter를 전체 영상에 반복적으로 적용함으로 변수의 수를 획기적으로 줄일 수 있으며, 항상성을 얻을 수 있음.

#### CNN의 구조 및 과정

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/75627787-c60d0780-5c16-11ea-8db6-329e879b8ca0.png" width="70%"></p>

    1. 특징을 추출
    2. topology 변화에 영향을 받지 않도록 함.
    3. 분류기 단계
    
 특징을 추출하기 위해 filter 계수를 값에 따라 각각 다른 특징을 얻을 수 있으며,  이 계수들은 특정 목적에 따라 고정이 되지만 CNN에서 사용하는 filter 혹은 convolutional layer는 학습을 통해 최적의 계수를 결정할 수 있게 하는 점이 다름.
 
 그리고 통상적인 sub-sampling은 보통 고정된 위치에 있는 픽셀을 고르거나, 혹은 sub-sampling 윈도우 안에 있는 픽셀들의 평균을 취함.
 이와 비슷하게 CNN에서는 pooling 방식(max, avg etc..)의 sub-sampling과정을 거침. 단, sub-sampling 과정은 feature map의 크기를 줄여주면서 topology invariance를 얻을 수 있음.
 
 convolution + sub-sampling 과정을 여러번 거치게 되면, global한 특징을 추출할 수 있다. 이렇게 얻어진 특징을 fully-connected network을 통해 학습을 시키기 되면, receptive field와 특성을 중점으로 topology변화에 인식 능력을 갖게 됨.
 
간략하게 아래 그림은 대략 전반적인 CNN의 구조를 설명함.

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/75656446-4c712a00-5ca7-11ea-8669-43ad0cebe07c.png" width="50%"></p> 

결과적으로 CNN은 여러 개의 layer로 구성이 되며, 주요 구성으로는 convolution layer, sub-sampling(pooling) layer, fully-connected layer로 입력 영상으로부터 주요 구성 layer를 통해, feature map을 만듬.

### 대표적인 CNN paper 두 개를 참고.
    
    1. Lecun - "Gradient-based learning applied to document recognition"
    2. Krizhevsky - "ImageNet classification with deep convolution neural network"
    
#### Lecun - "Gradient-based learning applied to document recognition"

LeCun은 CNN의 개념을 처음으로 만든 사람이며, 신경망 연구 정체기에 돌파구가 됨.

이는, CNN이 필기체 인식에 있어 탁월한 효과가 있음을 입증하였으며, 이것을 통해 다양한 분야에 적용되었음.

Lecun - CNN의 구조

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/75656789-13858500-5ca8-11ea-98c6-6910e613d8ba.png" width="50%"></p>

이 과정을 대략적으로 보면, convolution, sub-sampling(pooling)을 반복함으로써 feature map의 크기 줄어들면서 개수는 늘어나는 것을 볼 수 있으며, 그 후 FC(Fully connected) 을 연결하여 최종 class를 분류하는 것을 확인할 수 있음.

적용 결과 다양한 feature map을 얻을 수 있었으며, 그 조합을 통해 topology 변화에 강인한 특성을 갖게 되었으며 인식 능력이 크게 개선됨.

#### Krizhevsky - "ImageNet classification with deep convolution neural network"

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/75656804-1d0eed00-5ca8-11ea-9f14-1d40f3e8e36f.png" width="70%"></p>

위/아래 구조가 나눠져 있는 것은 2개의 gpu에 적용을 하기 위함.

총 5단계 convolution을 적용함.

    1단계 - 11 x 11 크기의 mask, stride = 4 
    => 55 x 55 크기 96개의 feature map을 얻음.

    2단계 - BatchNormalization과 max-pooling 거친 후 5 x 5 크기 mask
    => 27 x 27 크기 256개의 feature map을 얻음.

    3단계 - BatchNormalization과 max-pooling 거친 후 3 x 3 크기의 mask
    => 13 x 13 크기 384개의 feature map을 얻음.

    4단계 - max pooling 없이 3 x 3 크기 mask
    => 13 x 13 크기 384개의 feature map을 얻음.

    5단계 - 3 x 3 크기 mask
    => 13 x 13 크기 256개의 feature map을 얻음.

이렇게 단계별로 feature map을 확인할 수 있으며 간단하게 말하자면 feature map에서는 부드럽게 보이거나 에지 등을 추출할 수 있었다.

결과적으로 CNN 알고리즘을 통해 처리하고자 하는 과제에 따라 최종 convolution kernel의 계수가 달라질 수 있으며, 동일한 과제여도 학습에 사용하는 학습 데이터에 따라서도 달라질 수 있다(이는 설정한 하이퍼파라미터의 값에 따라서도 달라질 수 있다는 것을 의미). 그리고 계수의 값은 기존 신경망과 마찬가지로 gradient에 기반한 back-propagation에 의해 결정이 됨.
