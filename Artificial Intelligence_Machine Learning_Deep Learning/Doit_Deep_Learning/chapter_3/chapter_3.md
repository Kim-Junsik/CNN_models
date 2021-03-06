# 경사 하강법으로 학습하는 방법
* 여러가지 특성을 사용하면 높은 차원에서의 초평면을 그려야 한다
* 여러가지 초평면을 생각하는 것은 힘들며 따라서 특성을 1개 또는 2개를 사용하여 2차원 
* 또는 3차원의 그래프를 다룬다.
* 낮은 차원에서 얻은 직관은 높은 차원으로 확장이 가능한 경우가 있다.
* 선형 회귀의 목표는 x,y값이 주어지면 기울기와 절편을 구하는 것이다.
* 즉, 산점도 그래프를 잘 표현하는 직선의 방정식을 찾는 것이다.
* 경사하강법(gradient descent) : 모델이 데이터를 잘 표현할 수 있도록 기울기(변화량)을 사용하여
* 모델을 조금씩 조정하는 최적화 알고리즘
* ŷ = ωx + b --> a는 가중치를 의미하는 ω또는 계수를 의미하는 θ로 나타내며 y는 ŷ으로 나타낸다
* 가중치와 절편(ω, b)는 알고리즘이 찾은 규칙을 의미하며, ŷ은 예측값을 의미한다.

# 훈련데이터에 맞는 w와 b찾기
 1. 무작위로 w와 b를 정한다 (무작위로 모델 찾기)
 2. x에거 샘플을 하나꺼내 ŷ을 계산한다
 3. ŷ과 실제 y값을 비교한다
 4. ŷ이 y와 가까워지도록 w와 b를 조정한다
 5. 모든 샘플을 처리할 때까지 2~4반복한다
 
 # 변화율이 양수일때 가중치 업데이트 하는 방법
*  변화율이 0보다 큰 경우 y_hat이 증가하려면 w가 증가해야한다.
*  즉, 변화율이 양수인 점을 이용하여 변화율을 w에 더하는 방법으로 w를 증가시킬 수 있다.

# 변화율이 음수일떄 가중치 업데이트 하는 방법
*  w가 감소하면 h_hat은 증가한다
*  즉, 변화율이 음수인 점을 이용하여 변화율을 더하는 방법으로 y_hat을 증가시킬 수 있다.
* 두 방법의 문제점은 y_hat이 y에 비해 너무 작은 값이면 큰 폭으로 w와 b를 증가시킬 수 없으며,
* y_hat이 y보다 커지면 y_hat을 감소시키지 못한다

# 오차 역전파로 가중치와 절편 업데이트
* y에서 y_hat을 뺀 오차의 양을 곱하는 방법으로 w를 업데이트 한다면, y_hat이 y보다 많이 작을 경우
* 변화량의 크기를 크게 할 수 있고 y_hat이 y를 지나칠 경우 w와 b의 방향도 바꿀 수 있다.

# 경사하강법
* 어떤 손실함수(loss function)이 정의되있을때 손실함수의 값이 최소가 되는 지점을 찾는 방법

# 손실함수
* 예상한 값과 실제 타깃값의 차이를 함수로 정의한 것을 말한다
* 이전에 사용한 오차를 변화율에 곱하여 가중치와 절편을 업데이트 하는것은 '제곱 오차'라는 손실함수를 미분한 것이다.

# 제곱 오차(squared error)

* 타깃과 예측값을 뺀 다음 제곱한 것
* 제곱 오차가 최소가 되면 산점도 그래프를 가장 잘 표현한 직선이 그려진다
* 제곱 오차 함수의 최솟값을 알아내려면, 기울기에 따라 함수의 값이 낮은 쪽으로 이동해야 한다.
* 기울기를 구하려면 제곱 오차를 가중치나 절편에 대해 미분해야 한다.

* w에서 변화율을 더하지 않고 빼는 이유는 손실 함수의 낮은 쪽으로 이동하고 싶기 때문이다.
* 오차 역전파에서 적용하였던 수식(w + w_rate*err)은 제곱 오차를 미분한 것과 같다.

 * 제곱오차 식
  
    ![image](https://user-images.githubusercontent.com/46274774/83345687-26643000-a351-11ea-88f2-20a138b124f8.png)
  
 * 가중치의 미분및 적용
  
    ![image](https://user-images.githubusercontent.com/46274774/83345742-983c7980-a351-11ea-8055-21d24e6399a4.png)
  
    ![image](https://user-images.githubusercontent.com/46274774/83345765-be621980-a351-11ea-943f-f256bbe4cbe3.png)
  
 * 절편의 미분및 적용
  
   ![image](https://user-images.githubusercontent.com/46274774/83345812-1ac53900-a352-11ea-91e2-a59ab9994269.png)
  
    ![image](https://user-images.githubusercontent.com/46274774/83345820-34ff1700-a352-11ea-8447-7d8618cc67d3.png)
  
