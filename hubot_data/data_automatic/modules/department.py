from .base_data import department_info,contact_info,labs
from .question_patterns import question_patterns


def generate_department_qa():
        """department: 모든 질문 패턴에 대해 고정 답변 생성"""
        qa_pairs = []
        patterns = question_patterns["department"]
        for pattern in patterns:
            if "어떤 학과" in pattern:
                answer = department_info["교육목표"]
            elif "진로" in pattern:
                answer = department_info["진로"]
            elif "자격증" in pattern:
                answer = "정보통신공학과 학생들에게 추천하는 자격증으로는 정보처리기사, 네트워크관리사, 정보보안기사, CCNA/CCNP, AWS/Azure 클라우드 자격증 등이 있습니다."
            elif "과사무실 위치" in pattern:
                answer = f"정보통신공학과 과사무실은 {contact_info['과사무실']}에 위치해 있습니다."
            elif "전화번호" in pattern:
                answer = f"정보통신공학과 과사무실 전화번호는 {contact_info['과사무실연락처']}입니다."
            elif "실습실" in pattern:
                labs_info = ", ".join([lab["이름"] for lab in labs])
                answer = f"정보통신공학과에는 {labs_info} 등이 있습니다."
            else:
                answer = department_info["교육목표"]
            qa_pairs.append({"question": pattern, "answer": answer})
        return qa_pairs