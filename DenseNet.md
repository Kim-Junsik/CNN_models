## DenseNet(Densely Connected Convolutional Networks)

### -Abstract.

각 층을 feed-forward 방식으로 연결합니다. 

이전 레이어의 feature map을 계속해 다음 레이어의 입력과 연결하는 식인데, 이때 연결은 feature map끼리 더하기가 아닌 Concatenation을 시키는 방식입니다. 

이러한 구조를 통해 우리는 Vanishing Gradient 개선, Feature Propagation 강화, Feature Reuse, Parameter 수 절약이라는 이점을 얻을 수 있습니다.

### -Introducion

층이 점점 더 깊어지는 가운데 새롭게 대두 된 문제는, Input이나 gradient에 대한 정보가 여러 layer를 통과하는 경우 네트워크 양 끝에 도달하는 시점에는 이 정보가 vanish 혹은 wash out 될 수 있다는 것이다. 최근 이러한 문제에 대해 최근 많은 연구가 있었는데 모두 선행 레이어에서 후 레이어로 향하는 short path를 만든다는 특징을 가졌고, 이러한 통찰력을 확장하는 아키텍쳐를 제안했습니다. 네트워크의 레이어간 information flow를 극대화하기 위해, feature-map size가 동일한 모든 레이어가 직접 연결되는 것입니다.

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/75620351-5160ab80-5bcb-11ea-88e6-7a7e00495607.png" width="70%"></p>

제안하는 구조에서 feature들은 서로 concatenation 하여 결합됩니다. l번째 층은 모든 선행 conv block의 feature-map들로 구성된 l개의 입력을 가지며, 각 feature-map은 모든 L- l개의 후속 레이어로 전달됩니다. 이것은 기존의 아키텍처에서 L개의 connection 대신 L(L+1)/2 개의 connection을 도입하는 것입니다. Dense connectivity pattern에서 중복되는 feature-map은 다시 학습할 필요가 없기 때문에, DenseNet은 기존의 CNN보다 적은 수의 parameter만 필요합니다.

### -Related Work

Highway Networks는 100개 이상의 layer로 이루어진 end-to-end 네트워크를 효과적으로 학습시키는 방법을 제공한 최초의 아키텍처 중 하나이고, Bypassing path(우회 경로)는 very deep network의 학습에서 핵심적인 요소인 것으로 여겨졌으며, ResNet에서도 이를 지지하는 연구 결과가 나타났습니다. 

DenseNet은 extrmly deep하거나 wide한 구조로부터 representational power를 끌어내는 대신, feature의 재사용을 통해 네트워크의 잠재력을 활용함으로써, 학습하기 쉬우면서도 효율적인 parameter를 가진 압축 모델을 만들었고, 이는 DenseNet과 ResNet간의 주요 차이점입니다. DenseNets는 다른 layer의 feature-map을 연결하는 Inception network에 비해, 더 간단하고 효율적이라는 것.

### -DenseNet

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/75620343-1d858600-5bcb-11ea-8b3e-a418212ebc2f.png" width="100%"></p>

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/75620341-09da1f80-5bcb-11ea-958d-39ef52ab12b2.png" width="100%"></p>

#### Growth rate
DenseNet과 기존의 네트워크 구조의 중요한 차이점은, very narrow layer(예. k=12)를 가질 수 있다는 것이다. 여기서 hyperparameter K를 네트워크의 growth rate라고 하고, 상대적으로 작은 growth rate 로도 state-of-the-art 성능을 얻기에 충분하다는 것을 보여준다.

이러한 효과는, 각 layer들이 block 내의 모든 이전 feature-map에 접근함에 따라, 네트워크의 “collective knowledge”에 접근 된다는 것이다. Feature-map을 네트워크의 global state로 볼 수 있으며, 각 layer는 각자의 k feature-map에 이 state를 추가하는 것. Growth rate는 각 global state에 기여하는 새로운 정보의 양을 조절. 한 번 쓰여진 global state는 네트워크의 어디에서나 접근 할 수 있으며, 기존의 네트워크 아키텍처와 달리 (Concatenate로 연결되어 있기 때문에) layer-to-layer로 복제할 필요가 없다.

#### Bottleneck layers.

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/75620813-cedaea80-5bd0-11ea-9ca6-e2b6e4b3b802.png" width="70%"></p>

각 3x3 convolution 전에 1x1 convolution을 bottleneck layer로 도입하여, 입력 feature-map의 개수를 줄이고 계산 효율을 향상시킬 수 있음을 알 수 있었다. 이 디자인은 DenseNet에 특히 효과적이며, 이러한 bottleneck layer를 이용한다. 즉, BN-ReLU-Conv(1x1)-BN-ReLU-Conv(3x3)으로 이루어진
을 이용하며, 이를 DenseNet-B라고 칭한다.

#### Compression

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/75620816-e6b26e80-5bd0-11ea-8040-3dd1202dc17a.png" width="50%"></p>

모델을 보다 소형으로 만들기 위해, transition layer에서 feature-map의 개수를 줄일 수 있으며 dense block이 m 개의 feature-map을 포함하는 경우 다음 transition layer에서 출력 feature-map을 [θmc]개가 생성된다. 여기서 0 <θ≤1은 compression factor라고 한다. (θ 가 1인 경우에는 transition layer의 특징맵 개수가 변경되지 않는다)

θ<1 인 DenseNet을 DenseNet-C라고 칭하며, 실험에서는 θ=0.5로 설정한다. 또한, bottleneck layer와 θ<1 인 transition layer를 모두 사용하는 모델은 DenseNet-BC라고 칭한다.

#### Implementation Details
ImageNet을 제외한 모든 dataset에 대한 실험에는, 각각 동일한 수의 layer를 가진 3개의 dense block으로 구성 된 DenseNet을 사용한다. 첫 번째 dense block에 들어가기 전, input image를 입력으로 하며, 16개(DenseNet-BC는 growth rate의 2배)의 feature-map을 출력으로 하는 convolution이 수행되며, Kernel size가 3x3인 conv layer의 경우에는 feature-map의 크기를 고정하기 위해 zero-padding을 사용한다. 연속되는 dense block 사이에는 1x1 convolution과 2x2 average pooling이 뒤따라오는 transition layer를 사용하고, 마지막 dense block의 끝에는 global average pooling 후에 softmax classifier가 뒤따른다. 3개의 dense block에서 feature-map 크기는 각각 32x32, 16x16, 8x8이다.

실험에서 사용한 configuration은 다음과 같다.

    Basic DenseNet 
    L=40, k=12 
    L=100, k=12 
    L=100, k=24

    DenseNet-BC
    L=100, k=12 
    L=250, k=24 
    L=190, k=40

>ImageNet에 사용된 정확한 네트워크 구성은 아래 표 참조

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/75620789-92a78a00-5bd0-11ea-8978-e702c6360f82.png" width="50%"></p>

### - Experiments
Datasets : 두 개의 CIFAR dataset과 SVHN, ImageNet
Training : CIFAR와 SVHN은 Batch size를 64로 하고, 각각 300 / 40회의 epoch 동안 학습을 진행,
ImageNet은 Batch size를 256으로 하고, 90회의 epoch 동안 학습 진행 다양한 depth L과 growth k에 대해 실험했으며, 주요 결과는 아래 표를 참조하자

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/75620796-a9e67780-5bd0-11ea-90f6-54126a1ad061.png" width="50%"></p>

### - Discussion
표면적으로 DenseNet은 ResNet과 매우 유사하지만 입력을 Concatenation 함으로써, 모든 DenseNet layer에서 학습된 feature-map을 모든 후속 layer에서 접근 할 수 있게 되는 것이 다르다. 아래 그림에서는 DenseNet의 모든 변형과, 유사 성능의 ResNet의 parameter efficiency를 비교한 실험 결과를 보여준다.
 
<p align="center"><img src="https://user-images.githubusercontent.com/45933225/75620804-b965c080-5bd0-11ea-9120-6482b3b46bbb.png" width="70%"></p>

결과를 요약하면 다음과 같다. DenseNet의 학습 설정은 이전 섹션과 동일하게 유지했을 때, DenseNet-BC가 DenseNet의 변형 중 parameter efficiency가 가장 좋고, DenseNet-BC이 ResNet과 유사한 성능을 달성하는데 필요한 parameter는 1/3에 불과하다
 
### -Conclusion

이 논문에서는 새로운 convolutional network architecture인 Dense Convolutional Network(DenseNet)를 제안한다. 이 네트워크는 동일한 feature-map size를 가진 두 layer 사이에 direct connection을 도입, DenseNet은 자연스럽게 수백 개의 layer로 확장되는 반면, optimization difficulty는 없음을 보여줬다.
