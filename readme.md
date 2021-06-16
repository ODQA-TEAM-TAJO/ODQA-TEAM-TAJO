<p align="center"><img src="https://user-images.githubusercontent.com/50580028/121635097-a377c880-cac0-11eb-934b-8433220c87d2.png"></p>
<h1 align="center">TAJO chatbot📚</h1>

## 데모 영상

## Table of Contents
- [프로젝트 소개 ✨](#프로젝트-소개-)
- [요구 사항 🚀](#요구-사항-)
- [파일 구성](#파일-구성)
- [Retriever 학습 및 평가](#Retriever-학습-및-평가)
- [Reader(MRC) 학습 및 평가](#ReaderMRC-학습-및-평가)
- [Inference](#Inference) 
- [Contributor](#Contributor)

## 프로젝트 소개 ✨
#### Open Domain Question Answering
- 주어진 질문에 적절한 답을 도출하는 질의응답 모델 구현
- Retriever, Reader 두 단계로 구성
  <p align="left"><img src="https://user-images.githubusercontent.com/50580028/121690356-b611f200-cb00-11eb-8d45-3a96c87b6a01.png" width="70%" height="70%"></p>
- 프로젝트 진행 과정에 대해 자세히 알고 싶다면? 
  노션 링크 추가
## 요구 사항 🚀
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

# Haystack 설치
!pip install git+https://github.com/deepset-ai/haystack.git
```
## 파일 구성
```
$> tree -d
.
├── /dpr
│     ├── dpr_train.py
│     ├── korquad_preprocess.ipynb
│     └── squad_to_dpr.py
├── /mrc
│     ├── /tag_compound
│     │      ├── inference.py
│     │      ├── load_data.py
│     │      ├── tag_compound.py
│     │      ├── tag_inference.py
│     │      └── train.py
│     ├── arguments.py
│     ├── customAddedConv.py
│     ├── train.py
│     ├── trainer_qa.py
│     └── utils_qa.py
└── readme.md
``` 

## Retriever 학습 및 평가
### 데이터
- Retriver 학습 데이터 : wikipedia기반 QA 데이터셋(KorQuAD 1.0, AI hub 기계 독해)
### 학습 및 추론
- [korquad_preprocess.ipynb](https://github.com/ODQA-TEAM-TAJO/ODQA-TEAM-TAJO/blob/main/dpr/korquad_preprocess.py) 실행
  - KorQuAD 1.0, AI hub 데이터 셋을 합치고 DPR 학습에 맞도록 positive-negative pair dataset 구성
  - 각 query마다 Elasticsearch를 이용해서 negative sample을 16개씩 생성
```
python dpr_train.py
```
## Reader(MRC) 학습 및 평가
### 데이터
- MRC 학습 데이터 : KLUE MRC Dataset  

### 학습 및 추론
train.py 를 실행하면 mrc 모델의 학습이 진행됩니다. 
```
# 학습 예시 (학습 중 validation 을 동시에 하려면 --do_eval 추가)
python train.py --output_dir [path to save trained model] --do_train
# 추론 예시
python train.py --model_name_or_path [path to load trained model] --do_eval
```
## Inference
```
```
## Contributor
| [김남혁_T1014](https://github.com/skaurl) | [서일_T1093](https://github.com/Usurper47) | [엄희준_T1122](https://github.com/eomheejun) | [우종빈_T1129](https://github.com/JongbinWoo) | [이보현_T1148](https://github.com/bonniehyeon) | [장보윤_T1178](https://github.com/dataminegames) |
| :----------: |  :--------:  |  :---------: |  :---------: | :---------: | :---------: |
| ![55614265](https://user-images.githubusercontent.com/55614265/116680602-d2d9e680-a9e6-11eb-9207-6b8ff06757f7.jpeg) | ![46472729](https://user-images.githubusercontent.com/55614265/116680614-d53c4080-a9e6-11eb-9f38-38e8896c8b9f.jpeg) | ![50470448](https://user-images.githubusercontent.com/55614265/116680649-e2592f80-a9e6-11eb-8f9e-631c15313c5d.png) | ![44800643](https://user-images.githubusercontent.com/55614265/116680664-e6854d00-a9e6-11eb-89b8-1c822f8ec5b9.jpeg) | ![50580028](https://user-images.githubusercontent.com/55614265/116680672-ea18d400-a9e6-11eb-8e7f-d6af940cd263.jpeg) | ![45453533](https://user-images.githubusercontent.com/55614265/116680729-fb61e080-a9e6-11eb-83ee-20539665565f.png) |
