import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def main():
    # 1) 모델/토크나이저 로드
    MODEL_NAME = "snunlp/KR-FinBert-SC"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)                                  # 모델 이동
    print(f"Using device: {device}")

    # 2) 금융 관련 mock 문장 정의
    sentences = [
        "삼성전자의 2분기 실적이 시장 기대치를 상회했다.",
        "세계 경기 침체 우려로 국내 증시가 약세를 보이고 있다.", 
        "현대자동차는 전기차 라인업 확대 계획을 발표했다.",
        "원/달러 환율 급등으로 수입 기업들의 비용 부담이 커졌다.",
        "카카오페이는 신규 서비스 출시 소식에 주가가 상승했다."
    ]

    # 3) 전처리 ‑ 토큰화
    batch = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    batch = {k: v.to(device) for k, v in batch.items()}

    # 4) 추론 & 소프트맥스 확률 계산
    with torch.no_grad():
        outputs = model(**batch)
        probs   = F.softmax(outputs.logits, dim=-1)   # shape = (batch, 3)

    probs = probs.cpu()

    # 8) 결과 출력
    id2label = model.config.id2label   # {0:'negative', 1:'neutral', 2:'positive'}
    for sent, p in zip(sentences, probs):
        label_id = p.argmax().item()
        print(f"{sent} → {id2label[label_id]} (score={p[label_id]:.4f})")

if __name__ == "__main__":
    main()