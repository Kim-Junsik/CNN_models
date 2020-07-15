## Evaluating image segmentation models.
###### Jeremy Jordan글 개인적으로 정리한 글 입니다. 문제가 될 경우 삭제하겠습니다.

표준 기계 학습 모델을 평가할 때 일반적으로 예측을 true positives, false positives, true negatives, false negatives 네 가지 범주로 분류함.

image segmentation 조밀한 예측 작업의 경우 무엇으로 간주되는지, 일반적으로 예측을 평가할 수 있는 방법을 명확하게 알 수는 없다.

그래서 semantic and instance segmentation 기법을 평가하는 일반적인 방법에 대해 알고자 함.

### Semantic segmentation
이미지의 각 픽셀의 클래스를 예측하는 것이다.

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/85674397-c7030100-b6ff-11ea-847f-ed09ec2d93ef.png" width="60%"></p>

예측 출력 형태를 보면 입력의 공간 해상도(너비와 높이)를 예측할 수 있는 클래스의 수와 동등한 채널 깊이와 일치함. 그래서 각 채널은 특정 클래스가 있는 영역을 레이블하는 이진 마스크로 구성됨.

### Intersection over Union
Jaccard Index라고도 하는 IoU(Intersection over Union) metric은 기본적으로 대상 마스크와 예측 출력 간의 겹침 백분율을 정량화하는 방법이다.  -   loss function와 밀접한 관련이 있음.

간단히 말하자면, target mask와 predict mask 사이의 공통 픽셀 수를 측정하는 것이다. 구분짓는 것은 두 마스크의 존재하는 총 픽셀 수이다.

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/85675689-e8b0b800-b700-11ea-9e65-cce119c53fbe.png" width="60%"></p>

그래서 다음 아래 이미지를 갖고 시각적인 예를 보이고자 함.

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/85676015-2e6d8080-b701-11ea-8bab-06c042f18232.png" width="60%"></p>

그래서 intersection (A∩B)은 target, predict mask 공통 픽셀 수에 해당하며 union (A∪B)은  두 마스크의 존재하는 총 픽셀 수에 해당하는 것으로 다음 아래 이미지와 같다.

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/85676080-3af1d900-b701-11ea-9ced-f8854720df29.png" width="60%"></p>

-numpy code

    intersection = np.logical_and(target, prediction)
    union = np.logical_or(target, prediction)
    iou_score = np.sum(intersection) / np.sum(union)

### Pixel Accuracy
semantic segmentation을 평가하기 위한 방법으로 이미지에서 정확하게 분류된 픽셀의 백분율을 확인하여 판단하는 것이다. 여기서 픽셀 정확도는 다음 아래와 같은 공식으로 구분 짓는다.

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/85678808-c9fff080-b703-11ea-9b35-1f27d9adcd72.png" width="60%"></p>

여기서 분자는 각 클래스에 대한 정보이며, 분모는 모든 클래스에 대한 정보이다. 그리고 클래스 별 픽셀 정확도를 고려할 때는 이진 마스크를 사용하여 평가하는 것을 동시에 확인할 수 있다.

그렇지만 이에 대한 방법은 이미지 내에서 클래스 표현이 작을 때 잘못된 결과를 받을 수 있다는 것을 참고해야 한다.

### Instance segmentation
Instance Segmentation model은 이미지에서 감지 된 각 객체를 설명하는 local Segmentation Mask 모음을 생성한다. 예를 들어 아래 이미지와 같이 확인 할 수 있음.

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/85724592-51168e00-b72f-11ea-951a-5eeddb97d02c.png" width="60%"></p>

### Calculating Precision
predict mask 모음을 평가하기 위해 각 예측 마스크를 주어진 입력에 대해 사용 가능한 각 대상 마스크와 비교한다.

- true positive는 prediction-target mask 쌍 일부 사전 정의된 임계값을 초과하는 IOU 점수를 가질 경우 관찰됨.
- false positive는 예측된 predicted object mask에 연관된 ground truth object mask가 없음을 나타냄.
- false negative는 gournd truth object mask에 관련 predicted object mask가 없음을 나타냄.

다음 대표적인 두 가지 예제에 대해 아래 이미지를 통해서 보여줌.

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/85726352-f4b46e00-b730-11ea-83f2-e88ac7e1d528.png" width="60%"></p>


#### Precision
ground truth에 대한 positive detections purity(순도)를 효과적으로 보여준다. 다음 아래 공식과 같이 나타낼 수 있음.

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/85727360-e74bb380-b731-11ea-9bb2-ac2ae508b2e8.png" width="60%"></p>

#### Recall
positive predictions에 대한 completeness(완전성)을 효과적으로 보여준다. 다음 아래 공식과 같이 나타낼 수 있음.

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/85727922-6640ec00-b732-11ea-974b-3c954ef63f4a.png" width="60%"></p>

그러나 모델 출력의 예측 및 호출을 계산하려면 positive detection을 구성하는 요소를 정의해야 한다.
이를 위해 각 prediction, target mask 쌍 사이의 IoU Score를 계산한 다음 정의된 임계값을 촉돠하는 IoU Score를 갖는 mask 쌍을 결정한다.

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/85728493-ef582300-b732-11ea-9597-21a2aa804246.png" width="60%"></p>

그러나 지정된 IoU 임계값에서 단일 precision, recall Score를 계산한다고해서 모델의 전체 precision-recall curve의 동작이 적절하게 설명되지는 않는다.
대신 average precision를 사용하여 precision-recall curve 아래 영역을 효과적으로 통합할 수 있다.

다음 precision-recall curve을 예로 아래와 같이 보여줌.

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/85729664-f7649280-b733-11ea-9e22-176057d564b5.png" width="60%"></p>

먼저, 주어진 r point의 precision가 r보다 큰 recall의 max precision로 curve을 조정한다.

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/85730730-dc465280-b734-11ea-88d6-0e9792d2c09c.png" width="60%"></p>

그런 다음 수치 적분으로 곡선 아래 면적을 간단히 계산한다. 이 방법은 recall value의 범위에 걸쳐 평균을 산출하는 기존 접근 방식을 대체함.

각 mask IoU에 따른 예측 임계값 때문에 precision-recall curve이 완벽한 recall로 확장되지 않을 가능성이 높다는 점을 유의해야 함.

예를 들어 Microsoft COCO challenge detection 작업에 대한 기본 metric은 IoU 임계값을 사용하여 average precision score를 0.5 ~ 0.95(0.05 increments)로 평가함.

<p align="center"><img src="https://user-images.githubusercontent.com/45933225/85731848-d309b580-b735-11ea-9034-8992cfd4decb.png" width="60%"></p>

다중 클래스 객체의 예측 문제인 경우, 이 값은 모든 클래스에 걸쳐 평균이 된다.
