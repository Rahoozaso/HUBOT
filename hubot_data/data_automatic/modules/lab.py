from .base_data import labs
from .question_patterns import question_patterns

def generate_lab_qa():
        """lab: 가능한 모든 패턴×실습실 조합 Q&A 생성"""
        qa_pairs = []
        patterns = question_patterns["lab"]
        for pattern in patterns:
            if "실습실" in pattern:
                labs_info = ", ".join([f"{lab['이름']}({lab['위치']})" for lab in labs])
                answer = f"정보통신공학과에는 {labs_info} 등이 있습니다."
                qa_pairs.append({"question": pattern, "answer": answer})
            for lab in labs:
                if lab["이름"] in pattern:
                    if "위치" in pattern:
                        answer = f"{lab['이름']}은(는) {lab['위치']}에 위치해 있습니다."
                    elif "기자재" in pattern:
                        answer = f"{lab['이름']}에는 {lab['기자재']}가 있습니다."
                    elif "어디" in pattern:
                        answer = f"{lab['이름']}은(는) {lab['위치']}에 위치해 있으며, {lab['용도']}입니다."
                    else:
                        answer = f"{lab['이름']}은(는) {lab['위치']}에 위치한 {lab['용도']}이며, {lab['기자재']}가 있습니다."
                    qa_pairs.append({"question": pattern, "answer": answer})
            if "컴퓨터실" in pattern:
                answer = f"정보통신공학과 학생들이 이용할 수 있는 컴퓨터실은 " + \
                         ", ".join([f"{lab['이름']}({lab['위치']}, {lab['기자재']})" for lab in labs]) + " 등이 있습니다."
                qa_pairs.append({"question": pattern, "answer": answer})
            if "이용 가능 시간" in pattern:
                answer = "정보통신실험실은 정보통신공학과 학생들이 일과시간 중 항상 이용할 수 있습니다. 다른 실습실은 수업이나 세미나가 없을 때 이용 가능하며, 자세한 사항은 과사무실에 문의하시기 바랍니다."
                qa_pairs.append({"question": pattern, "answer": answer})
        return qa_pairs