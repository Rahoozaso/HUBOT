import json
import os
from .curriculum import generate_curriculum_qa
from .graduation import generate_graduation_qa
from .lab import generate_lab_qa
from .scholarship import generate_scholarship_qa
from .academic import generate_academic_qa
from .department import generate_department_qa
from .casual import generate_casual_qa

def generate_dataset():
    dataset = {
        "univ_qa": [],
        "chatting": []
    }
    dataset["univ_qa"].extend(generate_curriculum_qa())
    dataset["univ_qa"].extend(generate_graduation_qa())
    dataset["univ_qa"].extend(generate_lab_qa())
    dataset["univ_qa"].extend(generate_scholarship_qa())
    dataset["univ_qa"].extend(generate_academic_qa())
    dataset["univ_qa"].extend(generate_department_qa())
    dataset["chatting"].extend(generate_casual_qa())
    return dataset

def save_dataset(dataset, filename="cur_data/hubot_dataset.json"):
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)
    print(f"데이터셋이 '{filename}'에 저장되었습니다.")
    print(f"총 {len(dataset['univ_qa'])} 개의 학교 관련 Q&A와 {len(dataset['chatting'])} 개의 일상 대화가 생성되었습니다.")
    print(f"전체 데이터셋 크기: {len(dataset['univ_qa']) + len(dataset['chatting'])} 개")
    return filename
