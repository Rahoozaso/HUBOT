from .base_data import curriculum, required_courses
from .question_patterns import question_patterns

def generate_curriculum_qa():
        """curriculum: 가능한 모든 패턴×학년×학기×과목 조합 Q&A 생성"""
        qa_pairs = []
        patterns = question_patterns["curriculum"]
        years = ["1", "2", "3", "4"]
        semesters = ["1", "2"]
        all_subjects = sorted({s for subjects in curriculum.values() for s in subjects})
        for pattern in patterns:
            if "{year}" in pattern and "{semester}" in pattern:
                for year in years:
                    for semester in semesters:
                        key = f"{year}-{semester}"
                        if key in curriculum:
                            subjects = ", ".join(curriculum[key])
                            question = pattern.format(year=year, semester=semester)
                            answer = f"{year}학년 {semester}학기에는 {subjects} 등의 과목을 수강합니다."
                            qa_pairs.append({"question": question, "answer": answer})
            elif "{subject}" in pattern:
                for subject in all_subjects:
                    question = pattern.format(subject=subject)
                    year_semester = None
                    for key, subjects in curriculum.items():
                        if subject in subjects:
                            year_semester = key
                            break
                    is_required = subject in required_courses
                    required_text = "필수과목" if is_required else "선택과목"
                    if year_semester:
                        year, semester = year_semester.split("-")
                        if "언제" in question:
                            answer = f"{subject}는 {year}학년 {semester}학기에 개설되어 있습니다."
                        elif "필수" in question:
                            answer = f"{subject}는 정보통신공학과의 {required_text}입니다."
                        else:
                            answer = f"{subject}는 {year}학년 {semester}학기에 개설되는 {required_text}입니다."
                        qa_pairs.append({"question": question, "answer": answer})
            elif "{year}" in pattern:
                for year in years:
                    question = pattern.format(year=year)
                    required_in_year = []
                    for subject in required_courses:
                        for key, subjects in curriculum.items():
                            if subject in subjects and key.startswith(f"{year}-"):
                                required_in_year.append(subject)
                    if required_in_year:
                        answer = f"{year}학년 필수과목은 {', '.join(required_in_year)} 등이 있습니다."
                    else:
                        answer = f"{year}학년에는 필수과목이 없습니다."
                    qa_pairs.append({"question": question, "answer": answer})
            elif "전공필수" in pattern:
                question = pattern
                answer = f"정보통신공학과 전공필수 과목은 {', '.join(required_courses)} 등 총 {len(required_courses)}개가 있습니다."
                qa_pairs.append({"question": question, "answer": answer})
        return qa_pairs