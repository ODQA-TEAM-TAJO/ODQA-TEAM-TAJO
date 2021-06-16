<p align="center"><img src="https://user-images.githubusercontent.com/50580028/121635097-a377c880-cac0-11eb-934b-8433220c87d2.png"></p>
<h1 align="center">DPR</h1>

## 설명
### 개요
- [Dense Passage Retriever](https://arxiv.org/abs/2004.04906)을 한국어 데이터 셋으로 학습
    - 학습 데이터
        - [KorQuAD 1.0](https://korquad.github.io/KorQuad%201.0/)
        - [AI hub 기계 독해](https://aihub.or.kr/aidata/86)
- ODQA 테스크의 open-source인 [haystack](https://github.com/deepset-ai/haystack) 활용 

### 학습 데이터 셋 구성
- [korquad_preprocess.ipynb](https://github.com/ODQA-TEAM-TAJO/ODQA-TEAM-TAJO/blob/main/dpr/korquad_preprocess.py), [squad_to_dpr.py](https://github.com/ODQA-TEAM-TAJO/ODQA-TEAM-TAJO/blob/main/dpr/squad_to_dpr.py)
- 각 query마다 elastic search를 이용해서 negative sample을 16개씩 생성한다.

### 학습
- Pretrained KoBERT 를 fine-tuning
