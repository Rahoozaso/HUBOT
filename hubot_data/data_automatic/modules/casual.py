from .base_data import casual_responses
from .question_patterns import question_patterns

def generate_casual_qa():
        qa_pairs = []
        patterns = question_patterns["casual"]
        for pattern in patterns:
            keyword = pattern.replace("hubot, ", "")
            for key, responses in casual_responses.items():
                if keyword in key:
                    for response in responses:
                        qa_pairs.append({"question": pattern, "answer": response})
                    break
            else:
                qa_pairs.append({"question": pattern, "answer": "안녕하세요! 무엇을 도와드릴까요?"})
        return qa_pairs
