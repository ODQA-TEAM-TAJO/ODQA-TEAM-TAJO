![image](https://user-images.githubusercontent.com/50580028/121635097-a377c880-cac0-11eb-934b-8433220c87d2.png)

# ODQA-TEAM-TAJO
## 소개
ODQA 기반 챗봇 생성 

### 요구 사항
```
!pip install datasets
!pip install transformers
!pip install rank_bm25
!pip install elasticsearch
!pip install pororo

# Mecab 설치
!sudo apt-get install g++ openjdk-8-jdk python3-dev python3-pip curl
!python3 -m pip install --upgrade pip
!python3 -m pip install konlpy
!sudo apt-get install curl git
!bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh)
```
### 파일 구성(tree 구조)

### 데이터
- MRC 학습 데이터 : KLUE MRC Dataset
- Retriver 도메인 데이터 : wikipedia  

## 훈련, 평가, 추론

### MRC 학습 및 평가
train.py 를 실행하면 mrc 모델의 학습이 진행됩니다. 
```
# 학습 예시 (학습 중 validation 을 동시에 하려면 --do_eval 추가)
python train.py --output_dir [path to save trained model] --do_train
# 평가 예시
python train.py --model_name_or_path [path to load trained model] --do_eval
```
### Retriever 

```
```
### inference

```
```
