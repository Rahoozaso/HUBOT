from .base_data import labs, academic_info
from .question_patterns import question_patterns



def generate_academic_qa():
        """academic: 모든 질문 패턴에 대해 고정 답변 생성"""
        qa_pairs = []
        patterns = question_patterns["academic"]
        for pattern in patterns:
            if "재수강" in pattern:
                answer = academic_info["재수강"]
            elif "수강신청 학점" in pattern:
                answer = academic_info["수강신청학점"]
            elif "수업" in pattern:
                answer = "정보통신공학과 수업은 이론 수업과 실습이 병행되며, 주요 과목들은 교수님의 강의와 조교의 실습 지도로 이루어집니다."
            elif "실습실" in pattern:
                labs_info = ", ".join([f"{lab['이름']}({lab['위치']})" for lab in labs])
                answer = f"정보통신공학과 실습실은 {labs_info}이 있습니다."
            elif "학점 이월" in pattern:
                answer = "동일 년도 1학기 최대 수강학점을 수강하지 않은 학점 수만큼 동일 년도 2학기에 최대 3학점까지 이월되어 신청할 수 있습니다. 단, 9학기 이상 재학생은 제외됩니다."
            elif "계절학기" in pattern:
                answer = "계절학기는 방학 중에 개설되며, 수강신청 일정은 학사일정에 따라 공지됩니다. 한 계절학기당 최대 6학점까지 수강 가능합니다."
            else:
                answer = "자세한 학사 정보는 학교 홈페이지의 학사정보 섹션을 참고하시거나, 학사종합지원센터로 문의하시기 바랍니다."
            qa_pairs.append({"question": pattern, "answer": answer})
        return qa_pairs