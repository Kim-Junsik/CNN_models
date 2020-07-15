## 신경망 알고리즘의 벡터화
* 넘파이, 머신러닝, 딥러닝 패키지들은 다차원 배열의 계산을 빠르게 수행할 수 있다. --> 벡터화(vectorization)된 연산
## 확률적 경사 하강법 
* 가중치를 1번 업데이트 할때 1개의 샘플을 사용, 손싱함수의 전역 최솟값을 불안정하게 찾는다.
## 배치 경사 하강법
* 가중치를 1번 업데이트할 때 전체 샘플을 사용하므로 손실함수의 전역 최솟값을 안정적으로 찾는다.
*  가중치를 1번 업데이트할 때 사용되는 데이터의 갯수가 많으므로 알고리즘 1번 수행당 계산 비용이 많이 든다.
* 배치 경사 하강법은 전체 샘플을 사용하여 가중치를 업데이트 하므로 손실 값이 안정적으로 감소한다.
* 가중치의 변화가 연속적이므로 손실값도 안정적으로 수렴한다.

## 2개 층을 가진 신경망
* 입력층에서 전달되는 특성이 각 뉴런에 모두 전달된다.
  
<p align="center"><img src="https://user-images.githubusercontent.com/46274774/84624134-c4442700-af1b-11ea-8eba-e39cb4dd0a09.png" width="25%"></p>
  
<p align="center"><img src="https://user-images.githubusercontent.com/46274774/84624373-3583da00-af1c-11ea-8228-20e0b6cb2cb4.png" width="40%"></p>

<p align="center"><img src="https://user-images.githubusercontent.com/46274774/84624416-4a606d80-af1c-11ea-9132-a343a6ebf025.png" width="40%"></p>

<p align="center"><img src="https://user-images.githubusercontent.com/46274774/84624586-a2976f80-af1c-11ea-9827-aafb0345294c.png" width="40%"></p>

* 위 식에서 기억할 점은 여러개의 뉴런을 사용함으로써 가중치가 1개의 열을 가진 벡터가 아닌 2차원 행렬이 되었다는 것이다.
* 가중치 행렬의 크기는 입력의 개수 * 출력의 개수로 생각하면 된다.
* 하나의 뉴런만 사용한 경우 출력의 개수가 1이므로 가중치는 열 벡터가 된다.
* 이를 샘플 전체에 대해 수식으로 나타내면 다음과 같다
  
<p align="center"><img src="https://user-images.githubusercontent.com/46274774/84624665-c5c21f00-af1c-11ea-9f4c-74b3cbc07a84.png" width="20%"></p>

## 출력을 하나로 모으기
* 이진 분류 문제는 각 뉴런에서 출려된 값(z<sub>1</sub>, z<sub>2</sub>, z<sub>3</sub>, ... , z<sub>n</sub>)을 하나의 뉴런으로 다시 모아야 한다.
* 여러 특성의 값을 각 뉴런에 통과시키면 여러개의 출력값(a<sub>1</sub>, a<sub>2</sub>, a<sub>3</sub>, ..., a<sub>n</sub>)이 나오며, 이 값들 다시 모아 이진 분류를 수행할 기준값(z)를 만드는 것이다.

<p align="center"><img src="https://user-images.githubusercontent.com/46274774/84624723-e5594780-af1c-11ea-9bfe-71ce190952bb.png" width="35%"></p>

<p align="center"><img src="https://user-images.githubusercontent.com/46274774/84624815-0b7ee780-af1d-11ea-90e2-01f076121b38.png" width="20%"></p>

<p align="center"><img src="https://user-images.githubusercontent.com/46274774/84624876-29e4e300-af1d-11ea-8827-5a4bdccb3c19.png" width="20%"></p>

<p align="center"><img src="https://user-images.githubusercontent.com/46274774/84624972-56006400-af1d-11ea-936c-6577517156f4.png" width="20%"></p>

## 은닉층이 추가된 신경망

<p align="center"><img src="https://user-images.githubusercontent.com/46274774/84625070-80522180-af1d-11ea-8261-a15d7dfbf4bf.png" width="35%"></p>

* 입력층의 경우 입력이 들어가는 층으로 층의 개수에 포함 시키지 않는다.
* 위 그림은 입력 값이출력 층으로 전달되기 전에 2개의 뉴런으로 구성된 은닉층을 통과한다.
* 절편은 종종 번거로움을 피하기 위해 표시하지 않는 경우가 많은데 각 뉴런의 계산에 포함된다는 점을 잊으면 안된다.
* 위 신경망에서 하나의 층을 행렬로, 하나의 뉴런을 행렬의 열로 생각하면 신경망과 행렬의 관계를 쉽게 이해할 수 있다.
  
## 다층 신경망 정리

<p align="center"><img src="https://user-images.githubusercontent.com/46274774/84625414-29008100-af1e-11ea-92b0-1b4e09c5a5ea.png" width="35%"></p>

* 활성화 함수는 각 층마다 다를 수 있지만, 한 층에서는 모두 같아야 한다.
  * 각 층에는 하나 이상의 뉴런을 가지며, 은닉층과 출력층에 있는 모든 뉴런에는 활성화 함수가 필요하며 문제에 맞는 활성화 함수를 사용해야 한다.
  * 단, 같은 층에 있는 뉴런은 모두 같은 활성화 함수를 사용해야 한다.
* 모든 뉴런이 모두 연결되어 있다면 이는 완전 연결 신경망이라고 한다.
  * 위 사진의 신경망은 입력층, 은닉층, 출력층이 모두 연결된 완전 연결(fully-connected)신경망이라고 한다.
  * 뉴런이 모두 연결되어 있는 층을 완전 연결층이라고 한다.
  * 완전 연결 신경망은 다층 퍼셉트론, 밀집연결 신경망(densely-connected), 피드-포워드 신경망(feed-forward)이라고 부른다

## 다층 신경망의 경사 하강법
* 신경망에서 경사 하강법을 적용하려면 W<sub>2</sub>와 b<sub>2</sub>, 그리고 W<sub>1</sub>와b<sub>1</sub>에 대한 손실 함수 L의 도함수를 구해야 한다.
* 미분의 순서는 출력층에서 은닉층 방향이며, 손실함수 L은 로지스틱 손실 함수이다.

### 가중치에 대한 손실 함수의 미분(출력층)

<p align="center"><img src="https://user-images.githubusercontent.com/46274774/84625738-d5dafe00-af1e-11ea-8d78-6fe43dbea8ac.png" width="30%"></p>

* A<sub>1</sub>의 첫 번째 열은 첫 번째 뉴런의 활성화 출력이다.
* 이 열을 각 샘플이 만든 오차와 곱한 다음 모두 더하면 첫 번째 뉴런에 대한 그레디언트 총 합이 된다.
* A<sub>1</sub>의 두 번째 열은 모든 샘플에 대한 두 번째 뉴런의 활성화 출력이므로 샘플의 오차와 곱하면 두 번째 뉴런에 대한 그레디언트 총합이 된다.
* 행렬의 구성을 보면 A<sub>1</sub>의 크기는 (m,2)이고 -(Y-A<sub>2</sub>)의 크기는 (m,1)로 A<sub>1</sub>을 전치하여 -(Y-A<sub>2</sub>)와 곱해야 한다.
* 현재 구한 그레디언트 행렬은 모든 샘플에 대한 그레디언트 총 합이므로 가중치 행렬을 업데이트 하기 위해서는 평균 그레디언트를 구해야 한다.
* 그후 적절한 learning rate를 곱한 후에 가중치 행렬 W<sub>2</sub>를 업데이트한다.

### 절편에 대한 손실 함수의 미분(출력층)

<p align="center"><img src="https://user-images.githubusercontent.com/46274774/84625917-22263e00-af1f-11ea-83d6-9f048be372ba.png" width="30%"></p>

* 위 식에서 구한 그레디언트도 전체 샘플의 개수로 나누어 평균 그래디언트를 구하면 된다.

### 가중치에 대한 손실 함수의 미분(은닉층)

<p align="center"><img src="https://user-images.githubusercontent.com/46274774/84626050-61ed2580-af1f-11ea-8ca7-97ae3e6c44ec.png" width="30%"></p>

* &part;Z<sub>1</sub>/&part;W<sub>1</sub> 은 Z<sub>1</sub>=XW<sub>1</sub> + b<sub>1</sub>이므로 X이다.
* &part;A<sub>1</sub>/&part;Z<sub>1</sub>는 시그모이드 함수에서 a(1-a)였으므로 이는 A<sub>1</sub>&odot;(1-A<sub>1</sub>)이다. 여기서 &odot;는 원소별 곱셉 기호이다.
* &part;Z<sub>2</sub>/&part;A<sub>1</sub>는 Z<sub>2</sub>=A<sub>1</sub>W<sub>2</sub> + b<sub>2</sub>이므로 A<sub>1</sub>에 대해 미분하면 출력층의 가중치 W<sub>2</sub>만 남는다.
* &part;L/&part;Z<sub>2</sub>는 -(Y-A<sub>2</sub>)이다.
* 위 도함수들을 모두 곱하는데 곱하는 순서는 신경망의 역방향 순서로 곱한다.
*  &part;L/&part;Z<sub>2</sub>*&part;Z<sub>2</sub>/&part;A<sub>1</sub> 의 두 도함수를 곱하는 것은 각 샘플이 만든 오차를 출력층에 있는 2개의 뉴런에 반영시킨다는 의미이다.
*  최종적으로 모든 도함수를 곱하여 구한 그래디언트 행렬은 전체 샘플에 대한 그레디언트 합이므로 전체 샘플 개수로 나누어 평균을 구해야 한다.

### 절편에 대한 손실 함수의 미분과 도함수의 곱
* 위 절편에 대한 계산과 동일하다.

## 미니 배치를 사용하여 모델 훈련
* 매 에폭하다 전체 데이터를 사용하는 것이 아니라 조금씩 나누어 정방향 계산을 수행하고, 그레디언트를 구하여 가중치를 업데이트한다.
* 미니 배치 경사하강법은 작게 나눈 미니 배치만큼 가중치를 업데이트 한다.
* 미니 배치의 크기는 보통 16, 32, 64, ... 등의 2의 배수를 사용한다.
* 만약 미니 배치가 1이라면 이는 확률적 경사 하강법이 되며, 전체 데이터라면 이는 배치 경사 하강법이 된다.
* 미니 배치의 크기가 작다면 손실 함수의 전역 최솟값을 찾아 가는 과정이 크게 흔들리는 모양이 될것이며, 반대로 미니 배치의 크기가 커지면 손실 함수의 전역 최솟값을 안정적으로 찾아갈 것이다.
* 여기서 중요한 점은 미니 배치의 최적값은 정해진 값이 아니라는 점이다.
