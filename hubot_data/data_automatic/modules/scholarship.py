from .base_data import scholarships
from .question_patterns import question_patterns


def generate_scholarship_qa():
        """scholarship: 가능한 모든 패턴×장학명 조합 Q&A 생성"""
        qa_pairs = []
        patterns = question_patterns["scholarship"]
        for pattern in patterns:
            if "{scholarship_name}" in pattern:
                for name, desc in scholarships.items():
                    question = pattern.format(scholarship_name=name)
                    answer = desc
                    qa_pairs.append({"question": question, "answer": answer})
            elif "종류" in pattern:
                answer = f"정보통신공학과 장학금으로는 {', '.join(scholarships.keys())} 등이 있습니다."
                qa_pairs.append({"question": pattern, "answer": answer})
            elif "신청은 어디서" in pattern:
                answer = "장학금 신청은 학생지원팀(031-330-4037~8)에서 할 수 있으며, 매 학기 초에 공지되는 신청 기간을 확인해야 합니다."
                qa_pairs.append({"question": pattern, "answer": answer})
            elif "학자금 대출" in pattern:
                answer = scholarships.get("학자금대출", "")
                qa_pairs.append({"question": pattern, "answer": answer})
            elif "성적장학금" in pattern:
                answer = scholarships.get("성적장학금", "")
                qa_pairs.append({"question": pattern, "answer": answer})
            else:
                answer = "장학금에 관한 자세한 정보는 학생지원팀(031-330-4037~8)이나 정보통신공학과 홈페이지를 참고하시기 바랍니다."
                qa_pairs.append({"question": pattern, "answer": answer})

        return qa_pairs