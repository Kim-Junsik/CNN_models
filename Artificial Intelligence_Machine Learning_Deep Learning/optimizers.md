<아래 내용은 SEBASTIAN RUDER의 글을 참고하여 작성한 내용입니다.>

## 딥러닝에서의 3가지 문제
1. 학습이 잘 안된다(underfitting)
2. 학습이 너무 느리다(slow)
3. 학습된 모델이 융통성이 없다(overfitting)

## 학습이 잘 안되는 경우
* 학습시 사용하는 Activation function은 Sigmoid함수로 이를 그래프로 나타내면 아래와 같다.

<p align="center"><img src="https://user-images.githubusercontent.com/46274774/83487569-36614880-a4e6-11ea-9840-937411d8fb5e.png" width="30%"></p>

* 위 이미지를 보면 Sigmoid는 원점에 가까울 수록 높은 기울기(Gradient)를 가지지만 원점에서 멀어질 수록 그 값이 0에 가까워 진다.
* 이는 역전파(Back propagation)에서 레이어가 깊어질 수록 가중치의 업데이트가 사라지는 Vanishing Gradient현상이 발생할 수 있으며, 이것 때문에 fitting이 안되는 underfitting문제가 발생한다.
* 이를 해결하기 위해 ReLU(Rectified Linear Unit activation function)함수를 사용한다.

<p align="center"><img src="https://user-images.githubusercontent.com/46274774/83487624-4d079f80-a4e6-11ea-99fd-5c5e48b48f32.png" width="30%"></p>

* ReLU함수는 위 그림에서 보듯이 양수의 값에서 미분한 기울기가 1인 함수이다.

## 학습이 너무 느림
* 이를 해결하기 위해 여러 옵티마이져(optimizers)들이 만들어졌다.

<p align="center"><img src="https://user-images.githubusercontent.com/46274774/83487752-922bd180-a4e6-11ea-9590-8458b17c93d8.png" width="50%"></p>

<p align="center"><img src="https://user-images.githubusercontent.com/46274774/83488070-2a29bb00-a4e7-11ea-9180-eb37f33994e1.gif" width="50%"></p>

<p align="center"><img src="https://user-images.githubusercontent.com/46274774/83488023-17af8180-a4e7-11ea-87f9-cf98ced8e745.gif" width="50%"></p>

<p align="center"><img src="https://user-images.githubusercontent.com/46274774/83488028-1aaa7200-a4e7-11ea-8f94-47a0dbad653c.gif" width="50%"></p>

### Gradient Descent
> * 네트워크의 parameter들을 &theta;라고 했을 때, 네트워크에서 내놓는 결과값과 실제 결과값 사이의 차이를 정의하는 함수 Loss function J(&theta;)의 값을 최소화하기 위해 기울기(gradient) &nabla;<sub>&theta;</sub>J(&theta;)를 이용하는 방법이다.
>  * Gradient Descent에서는 &theta; 에 대해 gradient의 반대 방향으로 일정 크기만큼 이동해내는 것을 반복하여 Loss function J(&theta;) 의 값을 최소화하는 &theta; 의 값을 찾는다.

<p align="center"><img src="https://user-images.githubusercontent.com/46274774/83488136-4cbbd400-a4e7-11ea-9fdb-78d258136794.png" width="30%"></p>

> * Loss Function을 계산할 때 전체 train set을 사용하는 것을 **Batch Gradient Descent** 라고 한다. 
> * 위 방법은 한번 step을 내딛을 때 전체 데이터에 대해 Loss Function을 계산해야 하므로 너무 많은 계산량이 필요하다.
> *  이를 해결하기 위해 보통은 **Stochastic Gradient Descent (SGD)** 라는 방법을 사용한다.

### SGD(Stochastic Gradient Descent)
> * loss function을 계산할 때 전체 데이터(batch) 대신 일부 조그마한 데이터의 모음(mini-batch)에 대해서만 loss function을 계산한다.
> * batch gradient descent 보다 다소 부정확할 수는 있지만, 훨씬 계산 속도가 빠르기 때문에 같은 시간에 더 많은 step을 갈 수 있으며 여러 번 반복할 경우 보통 batch의 결과와 유사한 결과로 수렴한다. 
> * SGD를 사용할 경우 Batch Gradient Descent에서 빠질 local minima에 빠지지 않고 더 좋은 방향으로 수렴할 가능성도 있다.

### Momentum
> * Momentum 방식은 Gradient Descent를 통해 이동하는 과정에 일종의 ‘관성’을 주는 것이다.
> * 현재 Gradient를 통해 이동하는 방향과는 별개로, 과거에 이동했던 방식을 기억하면서 그 방향으로 일정 정도를 추가적으로 이동하는 방식이다.
> * v<sub>t</sub>를 time step t에서의 이동 벡터라고 할 때, 다음과 같은 식으로 이동을 표현할 수 있다.

<p align="center"><img src="https://user-images.githubusercontent.com/46274774/83488365-b0de9800-a4e7-11ea-8c23-f500a09aef9d.png" width="30%"></p>

> * 위 식에서 &gamma;는 얼마나 momentum을 줄 것인지에 대한 momentum term으로서, 보통 0.9 정도의 값을 사용한다.
> * 과거에 얼마나 이동했는지에 대한 이동 항 v를 기억하고, 새로운 이동항을 구할 경우 과거에 이동했던 정도에 관성항만큼 곱해준 후 Gradient을 이용한 이동 step 항을 더해준다.
> * 이동항 v<sub>t</sub> 는 다음과 같은 방식으로 정리할 수 있어, Gradient들의 지수평균을 이용하여 이동한다고도 해석할 수 있다.

<p align="center"><img src="https://user-images.githubusercontent.com/46274774/83488437-ce136680-a4e7-11ea-9196-d1663f55af4f.png" width="50%"></p>

> * Momentum 방식을 사용할 경우 다음과 같이 자주 이동하는 방향에 관성이 걸리게 되고, 진동을 하더라도 중앙으로 가는 방향에 힘을 얻기 때문에 SGD에 비해 상대적으로 빠르게 이동할 수 있다.

<p align="center"><img src="https://user-images.githubusercontent.com/46274774/83488480-e4212700-a4e7-11ea-9524-0039e2537ddd.png" width="30%"><img src="https://user-images.githubusercontent.com/46274774/83488493-e97e7180-a4e7-11ea-9d07-8c950ae5931f.png" width="30%"></p>

<p align="center"><img src="https://user-images.githubusercontent.com/46274774/83488521-f307d980-a4e7-11ea-9942-0380eb84f986.png" width="30%"></p>

> * Momentum 방식을 이용할 경우 위의 그림과 같이 local minima를 빠져나오는 효과가 있을 것이라고도 기대할 수 있다.
> * SGD를 이용할 경우 좌측의 local minima에 빠지면 gradient가 0이 되어 이동할 수가 없지만, momentum 방식의 경우 기존에 이동했던 방향에 관성이 있어 이 local minima를 빠져나오고 더 좋은 minima로 이동할 것을 기대할 수 있게 된다.
> * momentum의 단점은 momentum 방식을 이용할 경우 기존의 변수들 θ 외에도 과거에 이동했던 양을 변수별로 저장해야하므로 변수에 대한 메모리가 기존의 두 배로 필요하게 된다.

### Nesterov Accelerated Gradient (NAG)

<p align="center"><img src="https://user-images.githubusercontent.com/46274774/83488700-424e0a00-a4e8-11ea-9153-f0f7d0de50bf.png" width="70%"></p>

> * Momentum 방식에서는 이동 벡터 v<sub>t</sub> 를 계산할 때 현재 위치에서의 gradient와 momentum step을 독립적으로 계산하고 합친다면, NAG에서는 momentum step을 먼저 고려하여, momentum step을 먼저 이동했다고 생각한 후 그 자리에서의 gradient를 구해서 gradient step을 이동한다.

<p align="center"><img src="https://user-images.githubusercontent.com/46274774/83488731-4bd77200-a4e8-11ea-9e8f-4250b14c49f6.png" width="30%"></p>

> * NAG를 이용할 경우 Momentum 방식에 비해 보다 효과적으로 이동할 수 있다.
> * Momentum 방식의 경우 멈춰야 할 시점에서도 관성에 의해 훨씬 멀리 갈수도 있다는 단점이 존재하는 반면, NAG 방식의 경우 일단 모멘텀으로 이동을 반정도 한 후 어떤 방식으로 이동해야할 지를 결정한다.
> * 따라서 Momentum 방식의 빠른 이동에 대한 이점은 누리면서도, 멈춰야 할 적절한 시점에서 제동을 거는 데에 훨씬 용이하다고 생각할 수 있을 것이다.

### Adagrad
> * Adagrad(Adaptive Gradient)는 변수들을 update할 때 각각의 변수마다 step size를 다르게 설정해서 이동하는 방식이다.
> * 이 알고리즘의 기본적인 아이디어는 ‘지금까지 많이 변화하지 않은 변수들은 step size를 크게 하고, 지금까지 많이 변화했던 변수들은 step size를 작게 하자’ 라는 것이다.
> * 자주 등장하거나 변화를 많이 한 변수들의 경우 optimum에 가까이 있을 확률이 높기 때문에 작은 크기로 이동하면서 세밀한 값을 조정하고, 적게 변화한 변수들은 optimum 값에 도달하기 위해서는 많이 이동해야할 확률이 높기 때문에 먼저 빠르게 loss 값을 줄이는 방향으로 이동하려는 방식이라고 생각할 수 있다.
> * 특히 word2vec이나 GloVe 같이 word representation을 학습시킬 경우 단어의 등장 확률에 따라 variable의 사용 비율이 확연하게 차이나기 때문에 Adagrad와 같은 학습 방식을 이용하면 훨씬 더 좋은 성능을 거둘 수 있을 것이다.

<p align="center"><img src="https://user-images.githubusercontent.com/46274774/83488819-6c073100-a4e8-11ea-9ef7-2bd03f0f897d.png" width="30%"></p>

> * Neural Network의 parameter가 k개라고 할 때, G<sub>t</sub>는 k차원 벡터로서 ‘time step t까지 각 변수가 이동한 gradient의 sum of squares’ 를 저장한다.
> * θ를 업데이트하는 상황에서는 기존 step size η에 G<sub>t</sub>의 루트값에 반비례한 크기로 이동을 진행하여, 지금까지 많이 변화한 변수일 수록 적게 이동하고 적게 변화한 변수일 수록 많이 이동하도록 한다.
> * 이 때 ϵ은 10−4 ~ 10−8 정도의 작은 값으로서 0으로 나누는 것을 방지하기 위한 작은 값이다.
> * 여기에서 G<sub>t</sub>를 업데이트하는 식에서 제곱은 element-wise 제곱을 의미하며, θ를 업데이트하는 식에서도 ⋅ 은 element-wise한 연산을 의미한다.
> * Adagrad를 사용하면 학습을 진행하면서 굳이 step size decay등을 신경써주지 않아도 된다는 장점이 있다.
> * 보통 adagrad에서 step size로는 0.01 정도를 사용한 뒤, 그 이후로는 바꾸지 않는다.
> * 반면, Adagrad에는 학습을 계속 진행하면 step size가 너무 줄어든다는 문제점이 있다.
> * G에는 계속 제곱한 값을 넣어주기 때문에 G의 값들은 계속해서 증가하기 때문에, 학습이 오래 진행될 경우 step size가 너무 작아져서 결국 거의 움직이지 않게 된다.
> * 이를 보완하여 고친 알고리즘이 RMSProp과 AdaDelta이다.

### RMSProp
> * RMSProp은 딥러닝의 대가 제프리 힌톤이 제안한 방법으로서, Adagrad의 단점을 해결하기 위한 방법이다.
> * Adagrad의 식에서 gradient의 제곱값을 더해나가면서 구한 G<sub>t</sub> 부분을 합이 아니라 지수평균으로 바꾸어서 대체한 방법이다.
> * 이렇게 대체를 할 경우 Adagrad처럼 G<sub>t</sub>가 무한정 커지지는 않으면서 최근 변화량의 변수간 상대적인 크기 차이는 유지할 수 있다.

<p align="center"><img src="https://user-images.githubusercontent.com/46274774/83488948-a4a70a80-a4e8-11ea-8244-3b46cdbb86ca.png" width="30%"></p>

### AdaDelta
> * AdaDelta (Adaptive Delta) 는 RMSProp과 유사하게 AdaGrad의 단점을 보완하기 위해 제안된 방법이다.
> * 과거의 모든 제곱 된 그라디언트를 축적하는 대신 Adadelta는 축적 된 과거의 그라데이션의 창을 고정 된 크기로 제한한다.
> * 이전의 제곱 된 그라디언트를 비효율적으로 저장하는 대신에, 그라디언트의 합계는 모든 과거 제곱 된 그라디언트의 부식 평균으로 재귀 적으로 정의된다.
> * 시간 단계 t에서의 이동 평균 E[g<sup>2</sup>]t는 이전 평균 및 현재 기울기에서만 (모멘텀 항과 유사하게 분수 γ에 따라) : 

<p align="center"><img src="https://user-images.githubusercontent.com/46274774/83595374-c235ac00-a59c-11ea-91a3-74e38ca2e113.png" width="30%"></p>

> * γ를 모멘텀 항과 비슷한 값 0.9로 설정했으며 명확성을 위해 이제 우리는 바닐라 SGD 업데이트를 매개 변수 업데이트 벡터 Δθt로 다시 작성한다.

<p align="center"><img src="https://user-images.githubusercontent.com/46274774/83595424-f5783b00-a59c-11ea-82f5-eb0d2f749e8e.png" width="30%"></p>

> * 이전에 파생 된 Adagrad의 매개 변수 업데이트 벡터는 다음과 같은 형식을 취한다.

<p align="center"><img src="https://user-images.githubusercontent.com/46274774/83595470-0d4fbf00-a59d-11ea-9f55-0c1f0b85ce43.png" width="30%"></p>

> * 우리는 이제 대각선 행렬 G<sub>t</sub>를 과거 제곱 된 기울기에 대한 감쇠 평균으로 대체합니다. E[g<sup>2</sup>]t :

<p align="center"><img src="https://user-images.githubusercontent.com/46274774/83595516-2b1d2400-a59d-11ea-8f83-8fc677506749.png" width="30%"></p>

> * 분모는 그라디언트의 평균 제곱근(RMS)오차 기준이므로,이를 단시간 기준으로 대체 할 수 있다.

<p align="center"><img src="https://user-images.githubusercontent.com/46274774/83595552-47b95c00-a59d-11ea-95a8-67579b4ac9bb.png" width="30%"></p>

> * 작성자는이 업데이트의 단위(SGD, momentum 또는 Adagrad)가 일치하지 않음을 알린다.
> * 즉, 업데이트의 매개 변수와 동일한 가상 단위가 있어야한다.
> * 이를 실현하기 위해 먼저 지수 기하 급수적으로 다른 평균을 정의한다.
> * 이번에는 기울기를 제곱하지 않고 매개 변수 업데이트를 제곱한다.

<p align="center"><img src="https://user-images.githubusercontent.com/46274774/83595577-67508480-a59d-11ea-951d-16e182a30356.png" width="30%"></p>

> * 매개 변수 업데이트의 제곱 평균 제곱근 오차는 다음과 같다.

<p align="center"><img src="https://user-images.githubusercontent.com/46274774/83595612-7c2d1800-a59d-11ea-892a-fcff22965453.png" width="30%"></p>

> * RMS [Δθ] t는 알려지지 않았으므로 이전 시간 단계까지 매개 변수 업데이트의 RMS로 근사한다.
> * RMS [Δθ] t-1로 이전 업데이트 규칙의 학습률 η을 대체하면 최종적으로 Adadelta 업데이트 규칙이 생성된다.

<p align="center"><img src="https://user-images.githubusercontent.com/46274774/83595654-9666f600-a59d-11ea-8717-a2ca9d5bf145.png" width="30%"></p>

> * Adadelta를 사용하면 업데이트 규칙에서 제거되었으므로 기본 학습 속도를 설정할 필요조차 없다.

### Adam(Adaptive Momentum Estimation)
> * Adam (Adaptive Moment Estimation)은 RMSProp과 Momentum 방식을 합친 것 같은 알고리즘이다.
> * momentum의 개념을 이용하지만, w<sub>ij</sub>에 대한 momentum인 v<sup>(t)</sup><sub>ij</sub>과 learning rate를 조절하는 g<sup>(t)</sup><sub>ij</sub>는 다음과 같이 지수 이동 평균으로 계산된다.

<p align="center"><img src="https://user-images.githubusercontent.com/46274774/83489597-9e655e00-a4e9-11ea-9ce8-0829dde3cf97.png" width="30%"></p>

> * 다만, Adam에서는 m과 v가 처음에 0으로 초기화되어 있기 때문에 학습의 초반부에서는 m<sub>t</sub>,v<sub>t</sub>가 0에 가깝게 bias 되어있을 것이라고 판단하여 이를 unbiased 하게 만들어주는 작업을 거친다.
> * 이 식들을 이용하여 바이어스 보정 된 1차 및 2차 모멘트 추정치를 계산함으로써 바이어스를 방해한다.

<p align="center"><img src="https://user-images.githubusercontent.com/46274774/83489623-a91ff300-a4e9-11ea-9e42-db63b27cbe78.png" width="30%"></p>

<p align="center"><img src="https://user-images.githubusercontent.com/46274774/83489640-afae6a80-a4e9-11ea-8f81-485f0704a10d.png" width="30%"></p>

> * 보통 β1 로는 0.9, β2로는 0.999, ϵ 으로는 10−8 정도의 값을 사용한다고 한다.

### AdaMax
> * AdaMax는 Adam 논문에서 extension으로 제안된 알고리즘이다.
> * Adam은 식 g<sup>(t)</sup><sub>ij</sub>와 같이 L<sub>2</sub> norm을 기반으로 learning rate를 조절한다.
> * AdaMax는 L<sub>2</sub> norm을 기반으로 learning rate를 조절하는 부분을 L<sub>p</sub> norm으로 확장시킨 알고리즘이다.
> * 한 가지 문제점은 p가 매우 클 경우, L<sub>p</sub> norm은 극단적인 값을 갖는 등 매우 불안정하다는 것이다.
> * 그러나 Adam 논문의 저자는 p가 무한대로 갈 때, 매우 간단하고 안정적인 알고리즘이 만들어지는 것을 다음과 같이 보여준다.
> * 먼저, Adam에서 learning rate를 조절하는 g<sup>(t)</sup><sub>ij</sub>는 AdaMax에서 다음과 같이 L<sub>p</sub> norm으로 확장된다.

<p align="center"><img src="https://user-images.githubusercontent.com/46274774/83489494-6e1dbf80-a4e9-11ea-978a-81460f8d9d0b.png" width="50%"></p>

> * AdaMax를 제안한 논문에서는 p가 무한대로 갈 때의 (g<sup>(t)</sup><sub>ij</sub>)<sup>1/p</sup>를 다음과 같이 G<sup>(t)</sup><sub>ij</sub>로 정의한다.

<p align="center"><img src="https://user-images.githubusercontent.com/46274774/83489507-74ac3700-a4e9-11ea-9ebf-696772115e13.png" width="30%"></p>

> * 그 다음, w<sub>ij</sub>를 다음과 같이 update한다.

<p align="center"><img src="https://user-images.githubusercontent.com/46274774/83489517-7a098180-a4e9-11ea-9f45-52cb2a5656ea.png" width="30%"></p>

> * 이 때 v<sup>(t)</sup><sub>ij</sub>는 Adam에서처럼 v_hat<sup>(t)</sup><sub>ij</sub>와 같이 정의되며, 일반적으로 β1과 β2는 각각 0.9와 0.999로 설정된다.
