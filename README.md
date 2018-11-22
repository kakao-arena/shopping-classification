# shopping-classification

`쇼핑몰 상품 카테고리 분류` 대회 참가자에게 제공되는 코드 베이스입니다. 전처리와 간단한 분류기 그리고 평가 코드로 구성되어 있습니다. (코드는 python2.7/3.5+, keras, tensorflow 기준으로 작성되었습니다.)

## UPDATED
  - 2018.11.22 python3가 호환됩니다. [PR](https://github.com/kakao-arena/shopping-classification/pull/3)

## 실행 방법

0. 데이터의 위치
    - 내려받은 데이터의 위치는 소스가 실행될 디렉토리의 상위 디렉토리로(../) 가정되어 있습니다.
    - 데이터 위치를 바꾸기 위해서는 각 소스의 상단에 정의된 경로를 바꿔주세요.
1. `python data.py make_db train`
    - 학습에 필요한 데이터셋을 생성합니다. (h5py 포맷) dev, test도 동일한 방식으로 생성할 수 있습니다.
    - 위 명령어를 수행하면 `train` 데이터의 80%는 학습, 20%는 평가로 사용되도록 데이터가 나뉩니다.
    - 이 명령어를 실행하기 전에 `python data.py build_y_vocab`으로 데이터 생성이 필요한데, 코드 레파지토리에 생성한 파일이 포함되어 다시 만들지 않아도 됩니다. 
      - Python 2는 `y_vocab.cPickle` 파일을 사용하고, Python 3은 `y_vocab.py3.cPickle` 파일을 사용합니다.
    - `config.json` 파일에 동시에 처리할 프로세스 수를 `num_workers`로 조절할 수 있습니다.
2. `python classifier.py train ./data/train ./model/train`
    - `./data/train`에 생성한 데이터셋으로 학습을 진행합니다.
    - 완성된 모델은 `./model/train`에 위치합니다.
3. `python classifier.py predict ./data/train ./model/train ./data/train/ dev predict.tsv`
    - 단계 1. 에서 `train`의 20%로 생성한 평가 데이터에 대해서 예측한 결과를 `predict.tsv`에 저장합니다.
4. `python evaluate.py evaluate predict.tsv ./data/train/data.h5py dev ./data/y_vocab.cPickle`
    - 예측한 결과에 대해 스코어를 계산합니다.
    - Python 3에서는 `y_vocab.cPickle` 파일 대신 `y_vocab.py3.cPickle` 파일을 사용하여야 합니다.


## 제출하기
1. `python data.py make_db dev ./data/dev --train_ratio=0.0`
    - `dev` 데이터셋 전체를 예측용도로 사용하기 위해 `train_ratio` 값을 0.0으로 설정합니다.
2. `python classifier.py predict ./data/train ./model/train ./data/dev/ dev baseline.predict.tsv`
    - 위 실행 방법에서 생성한 모델로 `dev` 데이터셋에 대한 예측 결과를 생성합니다.
3. 제출
    - baseline.predict.tsv 파일을 zip으로 압축한 후 카카오 아레나 홈페이지에 제출합니다.


## 로직 설명
카테고리를 계층 구분없이 "대>중>소>세"로 표현하여 데이터를 구성했습니다. 그 뒤에 간단한 선형 모델로 네트워크를 구성했는데, 텍스트 데이터를 정규화한 후 단어 빈도가 높은 순서로 N개의 워드와 그에 대한 빈도를 입력으로 받습니다. 워드는 임베딩되고, 빈도는 가중치로 작용하게 됩니다.


## 기타
- 코드 베이스를 실행하기 위해서는 데이터셋을 포함해 최소 450G 가량의 공간이 필요합니다.

## 테스트 가이드라인
학습데이터의 크기가 100GB 이상이므로 사용하는 장비에 따라서 설정 변경이 필요합니다. `config.json`에서 수정 가능한 설정 중에서 아래 항목들이 장비의 사양에 민감하게 영향을 받습니다.

    - train_data_list
    - chunk_size
    - num_workers
    - num_predict_workers


`train_data_list`는 학습에 사용할 데이터 목록입니다. 전체 9개의 파일이며, 만약 9개의 파일을 모두 사용하여 학습하기 어려운 경우는 이 파일 수를 줄일 경우 시간을 상당히 단축시킬 수 있습니다. 

`chunk_size`는 전처리 단계에서 저장하는 중간 파일의 사이즈에 영향을 줍니다. Out of Memory와 같은 에러가 날 경우 이 옵션을 줄일 경우 해소될 수 있습니다.

`num_workers`는 전처리 수행 시간과 관련이 있습니다. 장비의 코어수에 적합하게 수정하면 수행시간을 줄이는데 도움이 됩니다.

`num_predict_workers`는 예측 수행 시간과 관련이 있습니다. `num_workers`와 마찬가지로 장비의 코어수에 맞춰 적절히 수정하면 수행시간을 단축하는데 도움이 됩니다.


### Benchmark

다음은 MacbookPro 노트북(8GB 램, 8 논리 코어, python2, tensorflow-CPU-Only)에서 사용한 셋팅과 각 단계별로 소요된 시간입니다. 설정은 기본값을 그대로 사용했으며, train_data_list에 파일을 하나 사용했을 때와 모두(9개) 사용했을때의 결과입니다.

    - train_data_list 1개 파일
        - `python data.py make_db`: 수행 시간 16분 42초
        - `python classifier.py train`: 수행 시간 6분 41초 / epoch
    - train_data_list 9개 파일
        - `python data.py make_db`: 수행 시간 2시간 3분
        - `python classifier.py train`: 수행 시간 50분 / epoch

## 라이선스

This software is licensed under the Apache 2 license, quoted below.

Copyright 2018 Kakao Corp. http://www.kakaocorp.com

Licensed under the Apache License, Version 2.0 (the “License”); you may not use this project except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an “AS IS” BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
