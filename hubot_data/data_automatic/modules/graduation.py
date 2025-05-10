from .base_data import graduation_requirements, required_courses
from .question_patterns import question_patterns

def generate_graduation_qa():
        """graduation: 가능한 모든 질문패턴×학번 조합 Q&A 생성"""
        qa_pairs = []
        patterns = question_patterns["graduation"]
        years = list(graduation_requirements.keys())
        for pattern in patterns:
            if "{year}" in pattern:
                for year in years:
                    question = pattern.format(year=year)
                    req = graduation_requirements[year]
                    if "졸업요건" in question:
                        answer = req["기본"]
                    elif "이중전공" in question:
                        if year == "2006이전":
                            answer = f"{year} 정보통신공학과 이중전공(제2전공) 이수 학점은 전공 {req['제2전공']['전공']}학점, 제2전공 {req['제2전공']['제2전공']}학점, 교양 {req['제2전공']['교양']}학점, 총 {req['제2전공']['합계']}학점입니다."
                        else:
                            answer = f"{year} 정보통신공학과 이중전공 이수 학점은 전공 {req['이중전공']['전공']}학점, 이중전공 {req['이중전공']['이중전공']}학점, 교양 {req['이중전공']['교양']}학점, 총 {req['이중전공']['합계']}학점입니다."
                    elif "부전공" in question and "전공심화" not in question:
                        if year == "2006이전":
                            answer = f"{year} 정보통신공학과 부전공 이수 학점은 전공 {req['부전공']['전공']}학점, 부전공 {req['부전공']['부전공']}학점, 교양 {req['부전공']['교양']}학점, 총 {req['부전공']['합계']}학점입니다."
                        else:
                            answer = f"{year} 정보통신공학과 부전공 이수 학점은 전공 {req['부전공']['전공']}학점, 부전공 {req['부전공']['부전공']}학점, 교양 {req['부전공']['교양']}학점, 총 {req['부전공']['합계']}학점입니다."
                    elif "전공심화" in question:
                        if year != "2006이전":
                            answer = f"{year} 정보통신공학과 전공심화 이수 학점은 전공 {req['전공심화']['전공']}학점, 교양 {req['전공심화']['교양']}학점, 총 {req['전공심화']['합계']}학점입니다."
                        else:
                            continue
                    elif "졸업논문" in question:
                        answer = req["졸업논문"]
                    elif "몇 학점" in question or "총" in question:
                        if year == "2006이전":
                            answer = f"{year} 정보통신공학과 졸업 이수 학점은 총 {req['제1전공']['합계']}학점입니다."
                        else:
                            answer = f"{year} 정보통신공학과 졸업 이수 학점은 총 {req['이중전공']['합계']}학점입니다."
                    elif "전공필수" in question:
                        answer = f"정보통신공학과 전공필수 과목은 {', '.join(required_courses)} 등 총 {len(required_courses)}개가 있습니다."
                    elif "외국어인증" in question:
                        answer = f"{year} 학번의 외국어인증 기준은 {req['외국어인증']}입니다."
                    elif "학번" in question and "졸업학점" in question:
                        if year == "2006이전":
                            answer = f"{year} 정보통신공학과 졸업 이수 학점은 총 {req['제1전공']['합계']}학점입니다."
                        else:
                            answer = f"{year} 학번 정보통신공학과 이중전공 이수학점은 {req['이중전공']['합계']}학점, 부전공 이수학점은 {req['부전공']['합계']}학점, 전공심화 이수학점은 {req['전공심화']['합계']}학점입니다."
                    else:
                        answer = req["기본"]
                    qa_pairs.append({"question": question, "answer": answer})
            else:
                question = pattern
                if "전공필수" in question:
                    answer = f"정보통신공학과 전공필수 과목은 {', '.join(required_courses)} 등 총 {len(required_courses)}개가 있습니다."
                else:
                    answer = "졸업요건 관련 정보는 학과 홈페이지를 참고하세요."
                qa_pairs.append({"question": question, "answer": answer})
        return qa_pairs
