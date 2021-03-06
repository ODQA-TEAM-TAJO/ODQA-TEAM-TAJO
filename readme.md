<p align="center"><img src="https://user-images.githubusercontent.com/50580028/121635097-a377c880-cac0-11eb-934b-8433220c87d2.png"></p>
<h1 align="center">TAJO chatbotπ</h1>

## Table of Contents
- [νλ‘μ νΈ μκ° β¨](#νλ‘μ νΈ-μκ°-)
- [μκ΅¬ μ¬ν­ π](#μκ΅¬-μ¬ν­-)
- [νμΌ κ΅¬μ±](#νμΌ-κ΅¬μ±)
- [Retriever νμ΅ λ° νκ°](#Retriever-νμ΅-λ°-νκ°)
- [Reader(MRC) νμ΅ λ° νκ°](#ReaderMRC-νμ΅-λ°-νκ°)
- [Inference](#Inference) 
- [Contributor](#Contributor)

## νλ‘μ νΈ μκ° β¨
#### Open Domain Question Answering
- ODQA λ μ£Όμ΄μ§ μ§λ¬Έμ μ μ ν λ΅μ μ°Ύλ κ³Όμ μ΄λ©°,Retriever, Reader λ λ¨κ³λ‘ κ΅¬μ±λμ΄ μμ΅λλ€.
- λ¨Όμ  μ§λ¬Έμ λν λ΅μ νκ³  μλ μ§λ¬Έμ κ²μνλ Retriver κ³Όμ μ κ±°μΉκ² λ©λλ€.
- μ΄ν ν΄λΉ μ§λ¬Έ μ μ λ΅μ μ°Ύλ Reader κ³Όμ μ ν΅ν΄ μ΅μ’ μμνμ μ»μ μ μμ΅λλ€.
  <p align="left"><img src="https://user-images.githubusercontent.com/50580028/121690356-b611f200-cb00-11eb-8d45-3a96c87b6a01.png" width="70%" height="70%"></p>
## μκ΅¬ μ¬ν­ π
```
!pip install datasets
!pip install transformers
!pip install elasticsearch
!pip install pororo

# Mecab μ€μΉ
!sudo apt-get install g++ openjdk-8-jdk python3-dev python3-pip curl
!python3 -m pip install --upgrade pip
!python3 -m pip install konlpy
!sudo apt-get install curl git
!bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh)

# Haystack μ€μΉ
!pip install git+https://github.com/deepset-ai/haystack.git
```
## νμΌ κ΅¬μ±
```
|-- dpr  # Retriever
|   |-- dpr_train.py
|   |-- korquad_preprocess.py
|   `-- squad_to_dpr.py
|-- mrc  # Reader
|   |-- tag_compound
|   |   |-- __init__.py
|   |   |-- data
|   |   |   |-- tag_train.tsv
|   |   |   |-- tag_train_augmented.tsv
|   |   |   `-- tag_valid.tsv
|   |   |-- inference.py
|   |   |-- load_data.py
|   |   |-- tag_compound.py
|   |   |-- tag_inference.py
|   |   `-- train.py
|   |-- arguments.py
|   |-- customAddedConv.py
|   |-- train.py
|   |-- trainer_qa.py
|   `-- utils_qa.py
`-- readme.md
```

## Retriever νμ΅ λ° νκ°
### λ°μ΄ν°
- Retriver νμ΅ λ°μ΄ν° : wikipediaκΈ°λ° QA λ°μ΄ν°μ(KorQuAD 1.0, AI hub κΈ°κ³ λν΄)

### νμ΅ λ° μΆλ‘ 
- [korquad_preprocess.ipynb](https://github.com/ODQA-TEAM-TAJO/ODQA-TEAM-TAJO/blob/main/dpr/korquad_preprocess.py) μ€ν
  - KorQuAD 1.0, AI hub λ°μ΄ν° μμ ν©μΉκ³  DPR νμ΅μ λ§λλ‘ positive-negative pair dataset κ΅¬μ±
  - κ° queryλ§λ€ Elasticsearchλ₯Ό μ΄μ©ν΄μ negative sampleμ 16κ°μ© μμ±
```
python dpr_train.py
```
## Reader(MRC) νμ΅ λ° νκ°
### λ°μ΄ν°
- MRC νμ΅ λ°μ΄ν° : KLUE MRC Dataset  

### νμ΅ λ° μΆλ‘ 
train.py λ₯Ό μ€ννλ©΄ mrc λͺ¨λΈμ νμ΅μ΄ μ§νλ©λλ€. 
```
cd ./mrc
# νμ΅ μμ (νμ΅ μ€ validation μ λμμ νλ €λ©΄ --do_eval μΆκ°)
python train.py --output_dir [path to save trained model] --do_train
# μΆλ‘  μμ
python train.py --model_name_or_path [path to load trained model] --do_eval
```
## Inference
[Demo-site](https://github.com/ODQA-TEAM-TAJO/ODQA-Demo-Site)

## Contributor
| [κΉλ¨ν_T1014](https://github.com/skaurl) | [μμΌ_T1093](https://github.com/Usurper47) | [μν¬μ€_T1122](https://github.com/eomheejun) | [μ°μ’λΉ_T1129](https://github.com/JongbinWoo) | [μ΄λ³΄ν_T1148](https://github.com/bonniehyeon) | [μ₯λ³΄μ€_T1178](https://github.com/dataminegames) |
| :----------: |  :--------:  |  :---------: |  :---------: | :---------: | :---------: |
| ![55614265](https://user-images.githubusercontent.com/55614265/116680602-d2d9e680-a9e6-11eb-9207-6b8ff06757f7.jpeg) | ![46472729](https://user-images.githubusercontent.com/55614265/116680614-d53c4080-a9e6-11eb-9f38-38e8896c8b9f.jpeg) | ![50470448](https://user-images.githubusercontent.com/55614265/116680649-e2592f80-a9e6-11eb-8f9e-631c15313c5d.png) | ![44800643](https://user-images.githubusercontent.com/55614265/116680664-e6854d00-a9e6-11eb-89b8-1c822f8ec5b9.jpeg) | ![50580028](https://user-images.githubusercontent.com/55614265/116680672-ea18d400-a9e6-11eb-8e7f-d6af940cd263.jpeg) | ![45453533](https://user-images.githubusercontent.com/55614265/116680729-fb61e080-a9e6-11eb-83ee-20539665565f.png) |
