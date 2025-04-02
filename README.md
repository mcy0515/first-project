# 텍스트 분석기 (Text Analyzer)

이 프로그램은 사용자가 직접 입력한 텍스트를 분석하고 요약하는 AI 기반 텍스트 분석 도구입니다.

## 주요 기능

- 직접 텍스트 입력 기능
- 텍스트 요약 생성
- 핵심 키워드 추출
- 감정 분석
- 키워드별 컨텍스트 설명
- 워드 클라우드 생성

## 설치 방법

다음 명령어로 필요한 라이브러리를 설치하세요:

```bash
pip install PyPDF2 nltk scikit-learn transformers pandas matplotlib wordcloud
```

## 사용 방법

1. `aidevelop.py` 파일을 실행합니다:

```bash
python aidevelop.py
```

2. 프로그램이 실행되면 분석할 텍스트를 입력하세요.
3. 텍스트 입력이 완료되면 빈 줄에서 엔터를 누르세요.
4. 분석이 완료되면 결과가 화면에 출력되고 `text_analysis.txt` 파일로 저장됩니다.
5. 워드 클라우드 이미지는 `wordcloud.png`로 저장됩니다.

## 분석 결과

분석 결과에는 다음 내용이 포함됩니다:

- 문서 요약
- 감정 분석 결과
- 주요 키워드 목록
- 각 키워드별 상세 설명
- 워드 클라우드 이미지 경로

## 주의사항

- 텍스트 요약은 영어 텍스트에 최적화되어 있습니다.
- 한글 텍스트 처리를 위해서는 추가 설정이 필요할 수 있습니다.
- 대용량 텍스트 입력 시 메모리 사용량이 증가할 수 있습니다.

## 라이센스

자유롭게 사용 가능합니다. 