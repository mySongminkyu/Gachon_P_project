from transformers import T5TokenizerFast, T5ForConditionalGeneration
import torch

# 1. 모델 & 토크나이저 불러오기
model_path = "/data/minkyu/P_project/pko-t5-small-corrector/checkpoint-261"

tokenizer = T5TokenizerFast.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path).to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

# 2. 예측 함수 정의
def correct_sentence(input_sentence):
    prefix = "correct: "
    inputs = tokenizer(prefix + input_sentence, return_tensors="pt", padding=True).to(model.device)
    outputs = model.generate(**inputs, max_length=64, num_beams=4)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 3. test
test_inputs = [
    "아빠 옷 사다 싶다",
    "동생 운동 하다 싶다",
    "나 과일 먹다 싶다",
    "밥 다음에 먹다 싶다"
]

for sent in test_inputs:
    corrected = correct_sentence(sent)
    print(f"입력: {sent} → 출력: {corrected}")
