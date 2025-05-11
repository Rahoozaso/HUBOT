import json
import os
import numpy as np
import torch
from transformers import BertModel, BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity

class HubotChatbot:
    def __init__(self, folder_path='./hubot_data/cur_data'):
        file_list = os.listdir(folder_path)
        if len(file_list) == 1:
            file_path = os.path.join(folder_path, file_list[0])
            with open(file_path, encoding='utf-8') as f:
                qa_dataset = json.load(f)
                self.univ_qa_list = qa_dataset["univ_qa"]
                self.chatting_qa_list = qa_dataset["chatting"]
        else:
            raise Exception("폴더에 파일이 없거나, 파일이 여러 개 있습니다.")

        self.tokenizer = BertTokenizer.from_pretrained('monologg/kobert')
        self.model = BertModel.from_pretrained('monologg/kobert')
        self.model.eval()

        self.univ_question_embeddings = [self.get_cls_embedding(item['question']) for item in self.univ_qa_list]
        self.chatting_question_embeddings = [self.get_cls_embedding(item['question']) for item in self.chatting_qa_list]

    #embedding 함수
    def get_cls_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=64)
        with torch.no_grad():
            outputs = self.model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :]
        return cls_embedding.squeeze(0).numpy()
    
    def answer(self, user_question, threshold=0.9):
        user_embedding = self.get_cls_embedding(user_question)
        univ_similarities = cosine_similarity([user_embedding], self.univ_question_embeddings)[0]
        chatting_similarities = cosine_similarity([user_embedding], self.chatting_question_embeddings)[0]

        univ_best_idx = int(np.argmax(univ_similarities))
        univ_best_score = univ_similarities[univ_best_idx]
        chatting_best_idx = int(np.argmax(chatting_similarities))
        chatting_best_score = chatting_similarities[chatting_best_idx]

        if max(univ_best_score, chatting_best_score) < threshold:
            return "죄송해요, 잘 모르겠어요.", None
        else:
            if univ_best_score >= chatting_best_score:
                return self.univ_qa_list[univ_best_idx]['answer'], "학교 QA"
            else:
                return self.chatting_qa_list[chatting_best_idx]['answer'], "일반 채팅"
