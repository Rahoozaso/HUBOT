from flask import Flask, request, jsonify
from hubot import HubotChatbot  # 위 클래스를 hubot_chatbot.py로 저장했다고 가정

app = Flask(__name__)
chatbot = HubotChatbot()

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_question = data.get('question')
    if not user_question:
        return jsonify({'error': '질문이 없습니다.'}), 400
    answer, category = chatbot.answer(user_question)
    return jsonify({'answer': answer, 'category': category})

if __name__ == '__main__':
    app.run(port=5000)
