# shopping-classification

`쇼핑몰 상품 카테고리 분류` 대회 참가자에게 제공되는 코드 베이스입니다. 전처리와 간단한 분류기 그리고 평가 코드로 구성되어 있습니다. (코드는 python2, keras, tensorflow 기준으로 작성되었습니다.)

## 실행 방법

0. 데이터의 위치
  - 내려받은 데이터의 위치는 소스가 실행될 디렉토리의 상위 디렉토리로(../) 가정되어 있습니다.
  - 데이터 위치를 바꾸기 위해서는 각 소스의 상단에 정의된 경로를 바꿔주세요.
1. `python data.py make_db train`
  -  학습에 필요한 데이터셋을 생성합니다. (h5py 포맷) dev, test도 동일한 방식으로 생성할 수 있습니다.
  - 이 명령어를 실행하기 전에 `python data.py build_y_vocab`으로 데이터 생성이 필요한데, 코드 레파지토리에 생성한 파일이 포함되어 다시 만들지 않아도 됩니다.
  - `config.json` 파일에 동시에 처리할 프로세스 수를 `num_workers`로 조절할 수 있습니다.
2. `python classifier.py train ./data/train ./model/train`
  - `./data/train`에 생성한 데이터셋으로 학습을 진행합니다.
  - 완성된 모델은 `./model/train`에 위치합니다.
3. `python classifier.py predict ./data/train ./model/train ./data/dev/ dev predict.tsv`
  - 1. 명령어로 `dev`에 대한 데이터셋을 만들고 위 명령어를 실행하면 예측 결과를 `predict.tsv`에 저장합니다.
4. `python evaluate.py evaluate predict.tsv ./data/dev/data.h5py dev ./data/y_vocab.cPickle`
  - 예측한 결과에 대해 스코어를 계산합니다.

## 로직 설명
카테고리를 계층 구분없이 "대>중>소>세"로 표현하여 데이터를 구성했습니다. 그 뒤에 간단한 선형 모델로 네트워크를 구성했는데, 텍스트 데이터를 정규화한 후 단어 빈도가 높은 순서로 N개의 워드와 그에 대한 빈도를 입력으로 받습니다. 워드는 임베딩되고, 빈도는 가중치로 작용하게 됩니다.
