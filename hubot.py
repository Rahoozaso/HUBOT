import json
import os
import numpy as np
import torch
from transformers import BertModel, BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity

#Kobert 사용
tokenizer = BertTokenizer.from_pretrained('monologg/kobert')
model = BertModel.from_pretrained('monologg/kobert')
model.eval()

#임베딩 함수
def get_cls_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] 토큰 벡터
    return cls_embedding.squeeze(0).numpy()

folder_path = './hubot_data/cur_data'  # 폴더 경로
file_list = os.listdir(folder_path)# 폴더 내 파일 목록 얻기

# 현재 사용 파일 하나 불러옴
if len(file_list) == 1:
    file_path = os.path.join(folder_path, file_list[0])
    with open(file_path, encoding='utf-8') as f:
        qa_dataset = json.load(f)
        #일반 채팅과 학과 qa로 나눔
        univ_qa_list = qa_dataset["univ_qa"]
        chatting_qa_list = qa_dataset["chatting"]
else:
    print("폴더에 파일이 없거나, 파일이 여러 개 있습니다.")#아니면 err
#각각 qa를 나누고 임베딩(단어 벡터 변환환)
univ_question_embeddings = [get_cls_embedding(item['question']) for item in univ_qa_list]
chatting_question_embeddings = [get_cls_embedding(item['question'])for item in chatting_qa_list]

user_question = input("질문을 입력하세요: ")
user_embedding = get_cls_embedding(user_question)  # 사용자 입력 임베딩
#각각의 유사도 검사(가장 유사한것)
univ_similarities = cosine_similarity([user_embedding], univ_question_embeddings)[0]
chatting_similarities = cosine_similarity([user_embedding], chatting_question_embeddings)[0]
#대학 qa 유사도 최댓값
univ_best_idx = int(np.argmax(univ_similarities))
univ_best_score = univ_similarities[univ_best_idx]
#채팅 qa 유사도 최댓값
chatting_best_idx = int(np.argmax(chatting_similarities))
chatting_best_score = chatting_similarities[chatting_best_idx]
print(chatting_best_score)
print(univ_best_score)
ambi = 0.9  # 모호값
if max(univ_best_score, chatting_best_score) < ambi:
    print("챗봇 답변: 죄송해요, 잘 모르겠어요.")
else:
    if univ_best_score >= chatting_best_score:
        best_answer = univ_qa_list[univ_best_idx]['answer']
        category = "학교 QA"
    else:
        best_answer = chatting_qa_list[chatting_best_idx]['answer']
        category = "일반 채팅"

    print(f"({category}) 챗봇 답변: {best_answer}")

