# 목표
$$f(x,y)=A*\sin(B*x)\sin(C*x)$$
위 u=f(x,y)라는 함수를 Task 집합체 그리고 A, B, C의 계수로 각각의 Task들이 다르다고 가정할 때,\n
Task간의 빠른 Transfer Learning을 하기전, Task 최적화된 모델을 생성하려한다.

해당 과정에서 Naive한 실험을 해보려 한다.


# 과정
## 간단한 기록

### 2023-11-21 금
* 데이터 생성
    * [make_dataset.ipynb](./make_dataset.ipynb)
* 단순 학습
    * [pre_training_withdata.ipynb](./pre_training_withdata.ipynb)
        * 단순 학습 과정에서 Data 없이 훈련을 진행하는 것에 문제가 많아 데이터를 추가하여 진행했다.
* 피드백
    * Data Loader 사용
        * 추후 더 많은 데이터 핸들링
    * 모델 initialize 고정하기
        * 특정 모델을 생성한 뒤 바로 저장하고 그 모델을 Load해서 사용하는 방식
### 2023-11-22

