# 다중 분류 신경망
## 소프트 맥스 함수(softmax function)

<p align="center"><img src="https://user-images.githubusercontent.com/46274774/84625414-29008100-af1e-11ea-92b0-1b4e09c5a5ea.png" width="35%"></p>

* 다중 분류 산경망을 만들기 위해 소프트 맥스(softmax)함수와 크로스 엔트로피(cross entropy)손실 함수를 사용한다.
* 이진 분류는 양성 클래스에 대한 확률 &ycirc;만 가지고 출력하면 됬지만 다중 분류는 클래스마다 확률 값을 출력한다.
* 다중 분류 산경망은 출력층에 분류할 클래스 개수만큼의 뉴런을 배치한다.
  
<p align="center"><img src="https://user-images.githubusercontent.com/46274774/84856523-e962a200-b0a1-11ea-9618-a155ae21d944.png" width="55%"></p>

* 만약 출력값이 80%, 70%, 50%라고 나올경우 이는 확실하게 어느 클래스인지 공정하게 비교하는 것이 어렵다.
* 따라서, 소프트 맥스 함수를 사용한다, 소프트 맥스 함수는 출력의 강도를 정규화한다. 즉 출력의 합을 1로 만든다.
* z<sub>i</sub>는 출력층의 각 뉴런에서 계산된 선형 출력을 의미한다.
* 다중 분류에서 출력층을 통과한 값들은 소프트맥스 함수를 거치며 적절한 확률값으로 변한다.

## 크로스 엔트로피 손실 함수(cross entropy loss function)

<p align="center"><img src="https://user-images.githubusercontent.com/46274774/84856562-05664380-b0a2-11ea-96e1-a51773210304.png" width="25%"></p>

* 크로스 엔트로피 손실 함수의 이진 분류 버전이 로지스틱 손실 함수이다. 
* 시그마 기호는 전체 클래스의 개수를 의미한다.
* 가운데 수식은 클래스마다 타깃과 활성화 출력의 로그값을 곱하여 더한것이다.
* 정답 클래스를 제외한 나머지 클래스의 p(x)의 값은 0이 되므로 이는 -log(a<sub>1</sub>)이 된다. (정답 클래스 p(x) = 1, 정답이 아닌 클래스 p(x) = 0, a<sub>x</sub>는 정답인 클래스의 뉴런의 활성화 출력값)
* 로지스틱 손실함수는 크로스 엔트로피 손실 함수에서 합(&Sigma;)기호를 빼고 -log(a), -log(1-a) 두개의 식으로 각각 양성클래스, 음성클래스에 대한 식이다.
* 이는 위애서본 로지스틱 함수와 같은 모양이다.

### 크로스 엔트로피 손실 함수의 미분
* 시그모이드와 달리 소프트맥스 함수는 출력 a<sub>1</sub>,a<sub>2</sub>,a<sub>3</sub>가 모두 z<sub>1</sub>의 함수이고, 손실함수 L이 a<sub>1</sub>,a<sub>2</sub>,a<sub>3</sub>에 대한 함수이므로 연쇠법칙은 다음과 같다.

<p align="center"><img src="https://user-images.githubusercontent.com/46274774/84856583-14e58c80-b0a2-11ea-8036-494ef07f8ec4.png" width="45%"></p>

<p align="center"><img src="https://user-images.githubusercontent.com/46274774/84856594-1ca53100-b0a2-11ea-96e2-bdb343bd86d3.png" width="45%"></p>

* 여기서 y<sub>1</sub>+y<sub>2</sub>+y<sub>3</sub>는 타깃의 합으로 이는 1이다.
* 즉 z<sub>1</sub>에대한 미분은 -(y<sub>1</sub>-a<sub>1</sub>), z<sub>2</sub>의 미분은 -(y<sub>2</sub>-a<sub>2</sub>), z<sub>3</sub>의 미분은 -(y<sub>3</sub>-z<sub>3</sub>)
* 즉 위 식을 벡터 z에 대해 정리하면 다음과 같다.

<p align="center"><img src="https://user-images.githubusercontent.com/46274774/84856603-229b1200-b0a2-11ea-9cf3-001c5217a6e0.png" width="25%"></p>

