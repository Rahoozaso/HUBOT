import json
import random
import os
from itertools import product

class HUFSICEDatasetGenerator:
    def __init__(self):
        # 기본 데이터 구조
        self.dataset = {
            "univ_qa": [],
            "chatting": []
        }
        
        # 교과과정 데이터 (2025학년도)
        self.curriculum = {
            "1-1": ["이산수학", "컴퓨터프로그래밍", "컴퓨팅사고", "AI공학개론", "마이크로영어", "대학영어1", "신입생세미나"],
            "1-2": ["컴퓨터프로그래밍및실습", "논리회로및실험", "선형대수", "확률과통계", "대학영어2"],
            "2-1": ["자료구조", "오픈소스소프트웨어", "AI정보이론", "회로이론의이해", "공업수학1", "정보통신개론"],
            "2-2": ["알고리즘설계와해석", "컴퓨터구조", "신호및시스템", "회로해석및실험", "공업수학2", "HUFS Career Vision Mentoring"],
            "3-1": ["머신러닝", "운영체제", "데이터통신", "통신공학", "IoT시스템", "인턴프로그래밍1"],
            "3-2": ["데이터베이스", "이산신호처리", "컴퓨터네트워크", "AI알고리즘", "정보통신기초종합설계", "인턴프로그래밍2"],
            "4-1": ["딥러닝", "네트워크보안", "통신네트워크설계", "데이터사이언스", "무선통신공학"],
            "4-2": ["소프트웨어공학", "이동통신", "멀티미디어통신", "디지털통신", "정보통신실종합설계및실습"]
        }
        
        # 전공필수 과목 (수강편람에서 추출)
        self.required_courses = [
            "컴퓨터프로그래밍", "논리회로및실험", "자료구조", "신호및시스템", 
            "데이터통신", "통신공학", "정보통신기초종합설계", "정보통신실종합설계및실습"
        ]
        
        # 학과 실습실 정보
        self.labs = [
            {
                "이름": "정보통신실험실",
                "위치": "공대 305호",
                "용도": "정보통신공학과 1,2,3 학년 실험실 (정보통신공학과 학생들은 일과시간중 항상 이용할 수 있습니다.)",
                "기자재": "컴퓨터 7대, 프린터 1대, 스캐너 1대"
            },
            {
                "이름": "정보통신 S/W 실험실",
                "위치": "공대 301호",
                "용도": "정보통신공학과 전용 강의실 및 실험실",
                "기자재": "컴퓨터 50대, 라우터, 빔 프로젝터, 디지털 통신 실험 키트, 함수 발생기, 오실로스코프, 로직 회로 실험기, 멀티미터, 마이크로 프로세서 실험 키트 등"
            },
            {
                "이름": "정보통신기초실험실",
                "위치": "공대 409호",
                "용도": "세미나실",
                "기자재": "컴퓨터 5대, 프린터 1대"
            }
        ]
        #과목 정보
        self.courses_info = {
            "자료구조": {
                "desc": "데이터를 효율적으로 저장하고 관리하는 방법과 다양한 자료구조의 원리를 배우는 과목입니다.",
                "credit": 3,
                "prerequisite": "컴퓨터프로그래밍",
                "required": True
            },
            "알고리즘설계와해석": {
                "desc": "알고리즘의 설계 원리와 효율성 분석 방법, 다양한 알고리즘 패러다임을 다루는 과목입니다.",
                "credit": 3,
                "prerequisite": "자료구조",
                "required": False
            },
            "컴퓨터프로그래밍": {
                "desc": "C 언어를 중심으로 프로그래밍의 기본 문법과 문제 해결 능력을 기르는 과목입니다.",
                "credit": 3,
                "prerequisite": None,
                "required": True
            },
            "이산수학": {
                "desc": "컴퓨터공학의 기초가 되는 논리, 집합, 그래프, 조합론 등 이산적 수학 개념을 배우는 과목입니다.",
                "credit": 3,
                "prerequisite": None,
                "required": True
            },
            "운영체제": {
                "desc": "컴퓨터 시스템의 운영 원리, 프로세스 관리, 메모리 관리, 파일 시스템 등을 배우는 과목입니다.",
                "credit": 3,
                "prerequisite": "자료구조",
                "required": True
            },
            "컴퓨터네트워크": {
                "desc": "컴퓨터 네트워크의 기본 개념, 프로토콜, 네트워크 계층 구조 등을 배우는 과목입니다.",
                "credit": 3,
                "prerequisite": "운영체제",
                "required": False
            },
            "확률과통계": {
                "desc": "확률, 통계, 데이터 분석의 기본 개념과 실습을 배우는 과목입니다.",
                "credit": 3,
                "prerequisite": None,
                "required": False
            },
            "선형대수": {
                "desc": "행렬, 벡터, 선형변환 등 선형대수의 기본 개념을 배우는 과목입니다.",
                "credit": 3,
                "prerequisite": None,
                "required": False
            },
            "머신러닝": {
                "desc": "데이터 기반 학습, 분류, 회귀, 신경망 등 머신러닝의 기초 이론과 응용을 배우는 과목입니다.",
                "credit": 3,
                "prerequisite": "선형대수, 확률과통계",
                "required": False
            }
        }


        # 학번별 졸업 요건
        self.graduation_requirements = {
            "2025이후": {
                "기본": "2025학번 이후 공과대학 졸업을 위해서는 전공, 교양, 기초 등 각 영역별 학점을 모두 이수하고 졸업학점 합계 이상을 취득해야 합니다.",
                "이중전공": {"전공": 42, "이중전공": 42, "부전공": 0, "기초": 6, "교양": 32, "합계": 126},
                "부전공": {"전공": 42, "이중전공": 0, "부전공": 21, "기초": 6, "교양": 32, "합계": 126},
                "전공심화부전공": {"전공": 54, "이중전공": 0, "부전공": 21, "기초": 6, "교양": 32, "합계": 126},
                "전공심화": {"전공": 54, "이중전공": 0, "부전공": 0, "기초": 6, "교양": 32, "합계": 126},
                "외국어인증": "FLEX 551점, TOEIC 645점, OPIc IM1 이상",
                "졸업논문": "필수 (졸업논문 또는 졸업시험 중 선택)"
            },
            "2023-2024": {
                "기본": "2023-2024학번 공과대학 졸업을 위해서는 전공, 교양 등 각 영역별 학점을 모두 이수하고 졸업학점 합계 이상을 취득해야 합니다.",
                "이중전공": {"전공": 57, "이중전공": 42, "부전공": 0, "교양": 32, "합계": 134},
                "부전공": {"전공": 66, "이중전공": 0, "부전공": 21, "교양": 32, "합계": 134},
                "전공심화부전공": {"전공": 75, "이중전공": 0, "부전공": 21, "교양": 32, "합계": 134},
                "전공심화": {"전공": 75, "이중전공": 0, "부전공": 0, "교양": 32, "합계": 134},
                "외국어인증": "FLEX 551점, TOEIC 645점, OPIc IM1 이상",
                "졸업논문": "필수 (졸업논문 또는 졸업시험 중 선택)"
            },
            "2015-2022": {
                "기본": "2015-2022학번 공과대학 졸업을 위해서는 전공, 교양 등 각 영역별 학점을 모두 이수하고 졸업학점 합계 이상을 취득해야 합니다.",
                "이중전공": {"전공": 57, "이중전공": 42, "부전공": 0, "교양": 32, "합계": 134},
                "부전공": {"전공": 66, "이중전공": 0, "부전공": 21, "교양": 32, "합계": 134},
                "전공심화부전공": {"전공": 75, "이중전공": 0, "부전공": 21, "교양": 32, "합계": 134},
                "전공심화": {"전공": 75, "이중전공": 0, "부전공": 0, "교양": 32, "합계": 134},
                "외국어인증": "FLEX 551점, TOEIC 645점, OPIc IM1 이상",
                "졸업논문": "필수 (졸업논문 또는 졸업시험 중 선택)"
            },
            "2007-2014": {
                "기본": "2007-2014학번 공과대학 졸업을 위해서는 전공, 교양 등 각 영역별 학점을 모두 이수하고 졸업학점 합계 이상을 취득해야 합니다.",
                "이중전공": {"전공": 54, "이중전공": 54, "부전공": 0, "교양": 26, "합계": 134},
                "부전공": {"전공": 54, "이중전공": 0, "부전공": 21, "교양": 26, "합계": 134},
                "전공심화부전공": {"전공": 75, "이중전공": 0, "부전공": 21, "교양": 26, "합계": 134},
                "전공심화": {"전공": 75, "이중전공": 0, "부전공": 0, "교양": 26, "합계": 134},
                "외국어인증": "FLEX 551점, TOEIC 650점",
                "졸업논문": "필수 (졸업논문 또는 졸업시험 중 선택)"
            },
            "2006이전": {
                "기본": "2006학번 이전 공과대학 졸업을 위해서는 전공, 교양 등 각 영역별 학점을 모두 이수하고 졸업학점 합계 이상을 취득해야 합니다.",
                "제1전공": {"전공": 54, "제2전공": 0, "부전공": 0, "교양": 28, "합계": 140},
                "제2전공": {"전공": 54, "제2전공": 42, "부전공": 0, "교양": 28, "합계": 140},
                "부전공": {"전공": 54, "제2전공": 0, "부전공": 21, "교양": 28, "합계": 140},
                "외국어인증": "FLEX 551점, TOEIC 650점",
                "졸업논문": "필수 (졸업논문 또는 졸업시험 중 선택)"
            }
        }
        
       
        # 학과 정보
        self.department_info = {
            "교육목표": "정보통신공학과는 4차 산업혁명의 핵심기술인 지능정보 기술, 정보보안 기술, 통신·네트워크 기술이 융합된 종합적인 학문입니다. 하드웨어와 소프트웨어 기술을 망라하여, 지능형시스템을 구축하고, 시스템 및 네트워크에서의 정보 보안을 보장하고, 멀티미디어 정보를 효율적으로 저장, 처리, 이용, 서비스하는 역량을 키우게 됩니다.",
            "진로": "삼성, LG, 현대 모비스 등 정보통신 제조업체, SKT, KT, 네이버, 카카오 등 정보통신 서비스 업체, 우리은행, KB, IBK 등 금융회사 전산직, 정보 교육(교사 자격증 취득 가능), 국책 및 기업 연구소 등 정보통신 관련 전 분야입니다."
        }
        
        # 학사 관련 정보
        self.academic_info = {
            "재수강": "재학 중 총 21학점(계절학기 제외)까지 재수강이 가능합니다. C+ 이하 과목만 재수강할 수 있으며, 성적 상한은 A0입니다.",
            "수강신청학점": "2021학번 이후 공과대학의 한 학기 최대 수강학점은 22학점입니다. 조기졸업자는 최대 3학점 추가 신청 가능합니다.",
            "수강신청일정": "1학년 2학기 수강신청 일정은, 예비수강신청함 이용 시작일은 1월 23일, 수강신청은 2월 6일, 수강신청 변경은 3월 4일~10일입니다."
        }
        
        # 장학금 정보
        self.scholarships = {
            "성적장학금": "총장 장학금(1명, 등록금 100%), 학장 장학금(8명, 40%), 학과장 장학금(8명, 30%)이 있으며, 직전 학기 14학점 이상, 평점 3.5 이상, 해당 학년 전공 3과목(9학점) 이상 수강해야 합니다.",
            "공로장학금": "7명, 정액 1,000,000원이 지급되며, 직전 학기 12학점 이상, 평점 2.0 이상이어야 합니다.",
            "면학장학금": "가정형편이 어려운 학생, 장애학생 등 등록금 지원이 필요한 학생에게 지급하는 장학금입니다.",
            "학자금대출": "취업후상환: 12학점, 평점 2.5 이상. 일반상환: 12학점, 평점 1.5 이상. 한국장학재단 사이트에서 신청합니다."
        }
        
        # 연락처 정보
        self.contact_info = {
            "학생지원팀": "031-330-4033",
            "장학팀": "031-330-4037",
            "학사종합지원센터": "031-330-4026",
            "과사무실": "공학관 5층",
            "과사무실연락처": "031-330-4740"
        }
        
        # 질문 패턴
        self.question_patterns = {
            "curriculum": [
                "hubot, {year}학년 {semester}학기에는 어떤 과목을 들어야 해?",
                "hubot, {subject} 과목은 언제 수강하는 게 좋아?",
                "hubot, {subject} 과목은 필수과목이야?",
                "hubot, {year}학년 필수과목은 뭐가 있어?",
                "hubot, 정보통신공학과 전공필수는 몇 개야?",
                "hubot, {year}학년 {semester}학기 전공과목 알려줘"
            ],
            "course_info": [
                "hubot, {course} 과목은 어떤 내용이야?",
                "hubot, {course}에서 뭘 배워?",
                "hubot, {course} 설명해줘.",
                "hubot, {course} 수업 내용 알려줘.",
                "hubot, {course}는 몇학점이야?",
                "hubot, {course} 몇 학점이야?",
                "hubot, {course}의 선행 과목은 어떤게 있어?",
                "hubot, {course} 듣기 전에 필요한 과목 있어?",
                "hubot, {course} 전공필수야?",
                "hubot, {course}는 전공필수 과목이야?",
                "hubot, {course}"
            ],
            "graduation": [
                "hubot, 정보통신공학과 졸업요건은 어떻게 돼?",
                "hubot, {year}학번 이중전공 이수 학점은 어떻게 되나요?",
                "hubot, {year}학번 부전공 이수학점이 궁금해요",
                "hubot, {year}학번 전공심화 총 몇 학점 이수해야 되나요?",
                "hubot, 졸업논문은 필수야?",
                "hubot, 정보통신공학과 졸업시험 과목은 어떻게 되나요?",
                "hubot, 졸업하려면 총 몇 학점을 이수해야 하나요?",
                "hubot, 전공필수 과목은 몇 개야?",
                "hubot, 외국어인증 기준은 어떻게 돼?",
                "hubot, {year}학번 졸업학점은 어떻게 돼?"
            ],
            "lab": [
                "hubot, 정보통신공학과 실습실은 어디에 있어?",
                "hubot, 정보통신실험실 위치가 어디야?",
                "hubot, 정보통신 S/W 실험실에는 무슨 기자재가 있어?",
                "hubot, 정보통신기초실험실은 어디에 있고 무엇을 하는 곳이야?",
                "hubot, 정보통신공학과 학생들이 이용할 수 있는 컴퓨터실은 어디에 있어?",
                "hubot, 학부 실습실 이용 가능 시간은 언제야?"
            ],
            "scholarship": [
                "hubot, {scholarship_name}은 어떤 조건이 있어?",
                "hubot, {scholarship_name} 신청 자격이 뭐야?",
                "hubot, 장학금 종류는 어떤 것들이 있어?",
                "hubot, 정보통신공학과 장학금 신청은 어디서 해?",
                "hubot, 성적장학금 기준 알려줄래?",
                "hubot, 학자금 대출 조건은 어떻게 돼?"
            ],
            "academic": [
                "hubot, 재수강 제도는 어떻게 돼?",
                "hubot, 한 학기 최대 수강신청 학점은?",
                "hubot, 정보통신공학과 수업은 어떤 식으로 진행돼?",
                "hubot, 정보통신공학과 실습실은 어디에 있어?",
                "hubot, 학점 이월 제도는 어떻게 돼?",
                "hubot, 계절학기는 어떻게 신청해?"
            ],
            "department": [
                "hubot, 정보통신공학과는 어떤 학과야?",
                "hubot, 정보통신공학과 졸업 후 진로는 어떻게 돼?",
                "hubot, 정보통신공학과 학생들이 취득하면 좋은 자격증은?",
                "hubot, 정보통신공학과 과사무실 위치가 어디야?",
                "hubot, 정보통신공학과 과사무실 전화번호가 뭐야?",
                "hubot, 정보통신공학과에는 어떤 실습실이 있어?"
            ],
            "casual": [
                "hubot, 안녕!",
                "hubot, 오늘 날씨 어때?",
                "hubot, 점심 뭐 먹었어?",
                "hubot, 이름이 뭐야?",
                "hubot, 취미가 뭐야?",
                "hubot, 잘 지내?",
                "hubot, 점심 추천해줘",
                "hubot, 오늘 기분이 안 좋아. 위로해줘",
                "hubot, 주말에 뭐하지?",
                "hubot, 너는 인공지능이야?"
            ]
        }
        
        # 일상 대화 응답
        self.casual_responses = {
            "안녕!": ["안녕하세요! 무엇을 도와드릴까요?", "반갑습니다! 어떤 도움이 필요하신가요?", "안녕하세요! 질문이 있으신가요?"],
            "오늘 날씨 어때?": ["오늘은 맑고 따뜻한 날씨입니다.", "오늘은 화창한 날씨에요. 외출하기 좋은 날이죠!", "현재 서울 기온은 25도로 따뜻합니다."],
            "점심 뭐 먹었어?": ["저는 인공지능이라 식사는 하지 않지만, 오늘의 급식 메뉴를 알려드릴 수 있어요!", "AI라서 먹지 않지만, 학생식당의 인기 메뉴는 돈까스입니다."],
            "이름이 뭐야?": ["저는 한국외대 정보통신공학과 챗봇 hubot입니다.", "안녕하세요, 저는 hubot이라고 합니다. 무엇을 도와드릴까요?"],
            "취미가 뭐야?": ["저는 학생들의 질문에 답변하는 것이 취미입니다. 여러분의 궁금증을 해결해 드리는 것이 가장 즐거워요!", "저는 AI라서 취미는 없지만, 학생들의 질문에 답변하는 것을 좋아합니다."],
            "잘 지내?": ["네, 저는 언제나 한결같이 잘 지내고 있어요. 여러분은 어떻게 지내시나요?", "저는 AI라서 항상 동일한 상태입니다. 여러분은 어떠신가요?"],
            "점심 추천해줘": ["오늘 점심으로는 학생식당의 오늘의 메뉴나 학교 근처 맛집을 추천드려요. 더 구체적인 추천이 필요하시면 말씀해주세요!", "학교 근처에 있는 김밥천국이나 찌개집이 인기가 많아요."],
            "오늘 기분이 안 좋아. 위로해줘": ["힘든 하루를 보내고 계시는군요. 잠시 휴식을 취하고 좋아하는 음악을 들으며 마음을 편안하게 가지면 도움이 될 거예요. 내일은 더 좋은 일이 있을 거예요!", "기분이 안 좋으시다니 유감이에요. 잠시 심호흡을 하고 좋아하는 일에 집중해보는 건 어떨까요?"],
            "주말에 뭐하지?": ["날씨가 좋다면 친구들과 캠퍼스 산책이나 근처 공원에 가보는 건 어떨까요? 또는 밀린 과제를 하거나 취미 활동을 즐기는 것도 좋은 방법이에요.", "도서관에서 공부하거나, 친구들과 만나거나, 집에서 쉬는 것도 좋은 방법입니다."],
            "너는 인공지능이야?": ["네, 저는 여러분의 학업과 대학 생활을 돕기 위해 설계된 인공지능 챗봇입니다.", "맞아요, 저는 한국외대 정보통신공학과의 AI 챗봇입니다."]
        }

    def generate_all_course_info_qa(self):
        """course_info: 가능한 모든 과목×질문패턴 조합 Q&A 생성"""
        qa_pairs = []
        subjects = list(self.courses_info.keys())
        patterns = self.question_patterns["course_info"]
        for subject, pattern in product(subjects, patterns):
            info = self.courses_info[subject]
            desc = info.get("desc", "")
            credit = info.get("credit", "정보 없음")
            prereq = info.get("prerequisite", None)
            required = info.get("required", False)
            required_str = "전공필수" if required else "전공선택"
            prereq_str = prereq if prereq else "없습니다."
            question = pattern.format(course=subject)
            if any(key in pattern for key in ["어떤 내용", "설명해줘", "수업 내용", "뭘 배워"]):
                answer = f"{subject}는 {desc} {credit}학점짜리 {required_str} 과목입니다. 선행과목은 {prereq_str}입니다."
            elif "몇학점" in pattern or "몇 학점" in pattern:
                answer = f"{subject}는 {credit}학점 과목입니다."
            elif "선행 과목" in pattern or "필요한 과목" in pattern:
                answer = f"{subject}의 선행과목은 {prereq_str}입니다."
            elif "전공필수" in pattern:
                answer = f"{subject}는 {required_str} 과목입니다."
            else:
                answer = f"{subject}에 대한 자세한 정보는 강의계획서를 참고하세요."
            qa_pairs.append({"question": question, "answer": answer})
        return qa_pairs

    def generate_all_curriculum_qa(self):
        """curriculum: 가능한 모든 패턴×학년×학기×과목 조합 Q&A 생성"""
        qa_pairs = []
        patterns = self.question_patterns["curriculum"]
        years = ["1", "2", "3", "4"]
        semesters = ["1", "2"]
        all_subjects = sorted({s for subjects in self.curriculum.values() for s in subjects})
        for pattern in patterns:
            if "{year}" in pattern and "{semester}" in pattern:
                for year in years:
                    for semester in semesters:
                        key = f"{year}-{semester}"
                        if key in self.curriculum:
                            subjects = ", ".join(self.curriculum[key])
                            question = pattern.format(year=year, semester=semester)
                            answer = f"{year}학년 {semester}학기에는 {subjects} 등의 과목을 수강합니다."
                            qa_pairs.append({"question": question, "answer": answer})
            elif "{subject}" in pattern:
                for subject in all_subjects:
                    question = pattern.format(subject=subject)
                    year_semester = None
                    for key, subjects in self.curriculum.items():
                        if subject in subjects:
                            year_semester = key
                            break
                    is_required = subject in self.required_courses
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
                    for subject in self.required_courses:
                        for key, subjects in self.curriculum.items():
                            if subject in subjects and key.startswith(f"{year}-"):
                                required_in_year.append(subject)
                    if required_in_year:
                        answer = f"{year}학년 필수과목은 {', '.join(required_in_year)} 등이 있습니다."
                    else:
                        answer = f"{year}학년에는 필수과목이 없습니다."
                    qa_pairs.append({"question": question, "answer": answer})
            elif "전공필수" in pattern:
                question = pattern
                answer = f"정보통신공학과 전공필수 과목은 {', '.join(self.required_courses)} 등 총 {len(self.required_courses)}개가 있습니다."
                qa_pairs.append({"question": question, "answer": answer})
        return qa_pairs

    def generate_all_graduation_qa(self):
        """graduation: 가능한 모든 질문패턴×학번 조합 Q&A 생성"""
        qa_pairs = []
        patterns = self.question_patterns["graduation"]
        years = list(self.graduation_requirements.keys())
        for pattern in patterns:
            if "{year}" in pattern:
                for year in years:
                    question = pattern.format(year=year)
                    req = self.graduation_requirements[year]
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
                        answer = f"정보통신공학과 전공필수 과목은 {', '.join(self.required_courses)} 등 총 {len(self.required_courses)}개가 있습니다."
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
                    answer = f"정보통신공학과 전공필수 과목은 {', '.join(self.required_courses)} 등 총 {len(self.required_courses)}개가 있습니다."
                else:
                    answer = "졸업요건 관련 정보는 학과 홈페이지를 참고하세요."
                qa_pairs.append({"question": question, "answer": answer})
        return qa_pairs

    def generate_all_lab_qa(self):
        """lab: 가능한 모든 패턴×실습실 조합 Q&A 생성"""
        qa_pairs = []
        patterns = self.question_patterns["lab"]
        for pattern in patterns:
            if "실습실" in pattern:
                labs_info = ", ".join([f"{lab['이름']}({lab['위치']})" for lab in self.labs])
                answer = f"정보통신공학과에는 {labs_info} 등이 있습니다."
                qa_pairs.append({"question": pattern, "answer": answer})
            for lab in self.labs:
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
                         ", ".join([f"{lab['이름']}({lab['위치']}, {lab['기자재']})" for lab in self.labs]) + " 등이 있습니다."
                qa_pairs.append({"question": pattern, "answer": answer})
            if "이용 가능 시간" in pattern:
                answer = "정보통신실험실은 정보통신공학과 학생들이 일과시간 중 항상 이용할 수 있습니다. 다른 실습실은 수업이나 세미나가 없을 때 이용 가능하며, 자세한 사항은 과사무실에 문의하시기 바랍니다."
                qa_pairs.append({"question": pattern, "answer": answer})
        return qa_pairs

    def generate_all_scholarship_qa(self):
        """scholarship: 가능한 모든 패턴×장학명 조합 Q&A 생성"""
        qa_pairs = []
        patterns = self.question_patterns["scholarship"]
        for pattern in patterns:
            if "{scholarship_name}" in pattern:
                for name, desc in self.scholarships.items():
                    question = pattern.format(scholarship_name=name)
                    answer = desc
                    qa_pairs.append({"question": question, "answer": answer})
            elif "종류" in pattern:
                answer = f"정보통신공학과 장학금으로는 {', '.join(self.scholarships.keys())} 등이 있습니다."
                qa_pairs.append({"question": pattern, "answer": answer})
            elif "신청은 어디서" in pattern:
                answer = "장학금 신청은 학생지원팀(031-330-4037~8)에서 할 수 있으며, 매 학기 초에 공지되는 신청 기간을 확인해야 합니다."
                qa_pairs.append({"question": pattern, "answer": answer})
            elif "학자금 대출" in pattern:
                answer = self.scholarships.get("학자금대출", "")
                qa_pairs.append({"question": pattern, "answer": answer})
            elif "성적장학금" in pattern:
                answer = self.scholarships.get("성적장학금", "")
                qa_pairs.append({"question": pattern, "answer": answer})
            else:
                answer = "장학금에 관한 자세한 정보는 학생지원팀(031-330-4037~8)이나 정보통신공학과 홈페이지를 참고하시기 바랍니다."
                qa_pairs.append({"question": pattern, "answer": answer})
        return qa_pairs

    def generate_all_academic_qa(self):
        """academic: 모든 질문 패턴에 대해 고정 답변 생성"""
        qa_pairs = []
        patterns = self.question_patterns["academic"]
        for pattern in patterns:
            if "재수강" in pattern:
                answer = self.academic_info["재수강"]
            elif "수강신청 학점" in pattern:
                answer = self.academic_info["수강신청학점"]
            elif "수업" in pattern:
                answer = "정보통신공학과 수업은 이론 수업과 실습이 병행되며, 주요 과목들은 교수님의 강의와 조교의 실습 지도로 이루어집니다."
            elif "실습실" in pattern:
                labs_info = ", ".join([f"{lab['이름']}({lab['위치']})" for lab in self.labs])
                answer = f"정보통신공학과 실습실은 {labs_info}이 있습니다."
            elif "학점 이월" in pattern:
                answer = "동일 년도 1학기 최대 수강학점을 수강하지 않은 학점 수만큼 동일 년도 2학기에 최대 3학점까지 이월되어 신청할 수 있습니다. 단, 9학기 이상 재학생은 제외됩니다."
            elif "계절학기" in pattern:
                answer = "계절학기는 방학 중에 개설되며, 수강신청 일정은 학사일정에 따라 공지됩니다. 한 계절학기당 최대 6학점까지 수강 가능합니다."
            else:
                answer = "자세한 학사 정보는 학교 홈페이지의 학사정보 섹션을 참고하시거나, 학사종합지원센터로 문의하시기 바랍니다."
            qa_pairs.append({"question": pattern, "answer": answer})
        return qa_pairs

    def generate_all_department_qa(self):
        """department: 모든 질문 패턴에 대해 고정 답변 생성"""
        qa_pairs = []
        patterns = self.question_patterns["department"]
        for pattern in patterns:
            if "어떤 학과" in pattern:
                answer = self.department_info["교육목표"]
            elif "진로" in pattern:
                answer = self.department_info["진로"]
            elif "자격증" in pattern:
                answer = "정보통신공학과 학생들에게 추천하는 자격증으로는 정보처리기사, 네트워크관리사, 정보보안기사, CCNA/CCNP, AWS/Azure 클라우드 자격증 등이 있습니다."
            elif "과사무실 위치" in pattern:
                answer = f"정보통신공학과 과사무실은 {self.contact_info['과사무실']}에 위치해 있습니다."
            elif "전화번호" in pattern:
                answer = f"정보통신공학과 과사무실 전화번호는 {self.contact_info['과사무실연락처']}입니다."
            elif "실습실" in pattern:
                labs_info = ", ".join([lab["이름"] for lab in self.labs])
                answer = f"정보통신공학과에는 {labs_info} 등이 있습니다."
            else:
                answer = self.department_info["교육목표"]
            qa_pairs.append({"question": pattern, "answer": answer})
        return qa_pairs

    def generate_all_casual_qa(self):
        """casual: 모든 질문패턴×모든 응답 조합 생성"""
        qa_pairs = []
        patterns = self.question_patterns["casual"]
        for pattern in patterns:
            keyword = pattern.replace("hubot, ", "")
            for key, responses in self.casual_responses.items():
                if keyword in key:
                    for response in responses:
                        qa_pairs.append({"question": pattern, "answer": response})
                    break
            else:
                qa_pairs.append({"question": pattern, "answer": "안녕하세요! 무엇을 도와드릴까요?"})
        return qa_pairs

    def generate_full_dataset(self):
        """모든 카테고리별 가능한 모든 Q&A 조합을 생성"""
        dataset = {
            "univ_qa": [],
            "chatting": []
        }
        dataset["univ_qa"].extend(self.generate_all_course_info_qa())
        dataset["univ_qa"].extend(self.generate_all_curriculum_qa())
        dataset["univ_qa"].extend(self.generate_all_graduation_qa())
        dataset["univ_qa"].extend(self.generate_all_lab_qa())
        dataset["univ_qa"].extend(self.generate_all_scholarship_qa())
        dataset["univ_qa"].extend(self.generate_all_academic_qa())
        dataset["univ_qa"].extend(self.generate_all_department_qa())
        dataset["chatting"].extend(self.generate_all_casual_qa())
        self.dataset = dataset
        return dataset

    def save_dataset(self, filename="hubot_ice_full_dataset.json"):
        """데이터셋을 JSON 파일로 저장"""
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.dataset, f, ensure_ascii=False, indent=4)
        print(f"데이터셋이 '{filename}'에 저장되었습니다.")
        print(f"총 {len(self.dataset['univ_qa'])} 개의 학교 관련 Q&A와 {len(self.dataset['chatting'])} 개의 일상 대화가 생성되었습니다.")
        print(f"전체 데이터셋 크기: {len(self.dataset['univ_qa']) + len(self.dataset['chatting'])} 개")
        return filename

if __name__ == "__main__":
    generator = HUFSICEDatasetGenerator()
    generator.generate_full_dataset()
    generator.save_dataset()
