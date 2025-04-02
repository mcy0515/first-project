import os
import re
import PyPDF2
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

try:
    # NLTK 데이터 다운로드
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class TextAnalyzer:
    def __init__(self):
        self.summarizer = pipeline("summarization")
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.question_generator = pipeline("text2text-generation", model="google/flan-t5-base")
    
    def load_text_from_file(self, file_path):
        """파일에서 텍스트를 로드하는 함수"""
        _, file_extension = os.path.splitext(file_path)
        
        if file_extension.lower() == '.pdf':
            return self._extract_text_from_pdf(file_path)
        elif file_extension.lower() in ['.txt', '.md', '.rtf']:
            return self._extract_text_from_text_file(file_path)
        else:
            raise ValueError(f"지원하지 않는 파일 형식입니다: {file_extension}")
    
    def _extract_text_from_pdf(self, pdf_path):
        """PDF 파일에서 텍스트 추출"""
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                text += pdf_reader.pages[page_num].extract_text()
        return text
    
    def _extract_text_from_text_file(self, text_path):
        """텍스트 파일에서 내용 추출"""
        with open(text_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    def preprocess_text(self, text):
        """텍스트 전처리: 소문자 변환, 특수문자 제거, 불용어 제거, 표제어 추출"""
        # 소문자 변환 및 특수문자 제거
        text = re.sub(r'[^\w\s]', '', text.lower())
        
        # 토큰화
        tokens = word_tokenize(text)
        
        # 불용어 제거 및 표제어 추출
        filtered_tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words and len(word) > 2]
        
        return filtered_tokens, " ".join(filtered_tokens)
    
    def extract_keywords(self, text, top_n=10):
        """TF-IDF를 사용하여 키워드 추출"""
        sentences = sent_tokenize(text)
        
        # TF-IDF 벡터화
        vectorizer = TfidfVectorizer(max_features=100)
        tfidf_matrix = vectorizer.fit_transform(sentences)
        
        # 단어별 TF-IDF 점수 계산
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.sum(axis=0).tolist()[0]
        
        # 단어와 점수를 매핑하고 정렬
        word_scores = {feature_names[i]: tfidf_scores[i] for i in range(len(feature_names))}
        sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_words[:top_n]
    
    def generate_summary(self, text, max_length=150, min_length=50):
        """텍스트 요약 생성"""
        # 텍스트가 너무 길면 처음 1000자만 사용
        if len(text) > 1000:
            text_for_summary = text[:1000]
        else:
            text_for_summary = text
            
        # Hugging Face 파이프라인으로 요약 생성
        summary = self.summarizer(text_for_summary, 
                                 max_length=max_length, 
                                 min_length=min_length, 
                                 do_sample=False)
        
        return summary[0]['summary_text']
    
    def analyze_sentiment(self, text):
        """텍스트의 감정 분석"""
        sentiment_analyzer = pipeline("sentiment-analysis")
        result = sentiment_analyzer(text[:512])  # 최대 입력 길이 제한
        return result[0]
    
    def explain_keyword(self, keyword, text, context_size=2):
        """키워드에 대한 컨텍스트 설명 생성"""
        sentences = sent_tokenize(text)
        keyword_contexts = []
        
        for i, sentence in enumerate(sentences):
            if keyword.lower() in sentence.lower():
                start = max(0, i - context_size)
                end = min(len(sentences), i + context_size + 1)
                context = " ".join(sentences[start:end])
                keyword_contexts.append(context)
        
        if not keyword_contexts:
            return f"'{keyword}'에 대한 설명을 찾을 수 없습니다."
        
        return keyword_contexts[0]
    
    def generate_word_cloud(self, text, output_path='wordcloud.png'):
        """워드 클라우드 이미지 생성"""
        tokens, _ = self.preprocess_text(text)
        word_freq = Counter(tokens)
        
        wordcloud = WordCloud(width=800, height=400, 
                             background_color='white', 
                             max_words=100).generate_from_frequencies(word_freq)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        return output_path
    
    def generate_exam_questions(self, summary, num_questions=5):
        """요약된 내용을 바탕으로 시험 문제 생성"""
        questions = []
        
        # 주관식 문제 생성
        for i in range(num_questions):
            prompt = f"Generate a question about the following text: {summary}"
            response = self.question_generator(prompt, max_length=100)
            question = response[0]['generated_text'].strip()
            
            # 답변 생성
            answer_prompt = f"Answer the following question based on this text: '{summary}'. Question: {question}"
            answer_response = self.question_generator(answer_prompt, max_length=150)
            answer = answer_response[0]['generated_text'].strip()
            
            questions.append({
                'question': question,
                'answer': answer,
                'type': '주관식'
            })
        
        return questions
    
    def analyze_text(self, text, keyword_count=10):
        """텍스트 종합 분석"""
        # 텍스트 전처리
        _, processed_text = self.preprocess_text(text)
        
        # 요약 생성
        summary = self.generate_summary(text)
        
        # 키워드 추출
        keywords = self.extract_keywords(text, keyword_count)
        
        # 감정 분석
        sentiment = self.analyze_sentiment(text)
        
        # 키워드별 설명 생성
        keyword_explanations = {}
        for keyword, _ in keywords:
            explanation = self.explain_keyword(keyword, text)
            keyword_explanations[keyword] = explanation
        
        # 워드 클라우드 생성
        wordcloud_path = self.generate_word_cloud(text)
        
        # 시험 문제 생성
        exam_questions = self.generate_exam_questions(summary)
        
        results = {
            'summary': summary,
            'keywords': keywords,
            'sentiment': sentiment,
            'keyword_explanations': keyword_explanations,
            'wordcloud_path': wordcloud_path,
            'exam_questions': exam_questions
        }
        
        return results
    
    def analyze_document(self, file_path, keyword_count=10):
        """문서 파일 종합 분석"""
        # 텍스트 로드
        text = self.load_text_from_file(file_path)
        
        # 텍스트 분석 수행
        results = self.analyze_text(text, keyword_count)
        
        # 파일명 추가
        results['filename'] = os.path.basename(file_path)
        
        return results
    
    def format_results(self, results):
        """분석 결과를 텍스트로 포맷팅"""
        filename = results.get('filename', '입력 텍스트')
        output = f"# 문서 분석 결과: {filename}\n\n"
        
        # 요약
        output += "## 요약\n"
        output += f"{results['summary']}\n\n"
        
        # 감정 분석
        output += "## 감정 분석\n"
        output += f"감정: {results['sentiment']['label']}, 확률: {results['sentiment']['score']:.2f}\n\n"
        
        # 키워드
        output += "## 주요 키워드\n"
        for i, (keyword, score) in enumerate(results['keywords'], 1):
            output += f"{i}. {keyword} (점수: {score:.4f})\n"
        output += "\n"
        
        # 키워드별 설명
        output += "## 키워드 설명\n"
        for keyword, explanation in results['keyword_explanations'].items():
            output += f"### {keyword}\n"
            output += f"{explanation}\n\n"
        
        # 시험 문제
        output += "## 시험 문제\n"
        for i, question in enumerate(results['exam_questions'], 1):
            output += f"### 문제 {i}\n"
            output += f"Q: {question['question']}\n"
            output += f"A: {question['answer']}\n\n"
        
        # 워드 클라우드
        output += f"## 워드 클라우드\n"
        output += f"워드 클라우드 이미지: {results['wordcloud_path']}\n"
        
        return output

# 사용 예시
def main():
    analyzer = TextAnalyzer()
    
    print("텍스트 분석기에 오신 것을 환영합니다.")
    print("\n분석할 텍스트를 입력하세요. 입력을 완료하려면 빈 줄에서 엔터를 누르세요:")
    
    try:
        # 텍스트 입력 받기
        lines = []
        while True:
            line = input()
            if not line:
                break
            lines.append(line)
        
        text = '\n'.join(lines)
        if not text.strip():
            print("텍스트가 입력되지 않았습니다.")
            return
            
        # 텍스트 분석
        results = analyzer.analyze_text(text)
        
        # 결과 출력
        formatted_results = analyzer.format_results(results)
        print(formatted_results)
        
        # 결과 파일로 저장
        output_file = "text_analysis.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(formatted_results)
        
        print(f"\n분석 결과가 {output_file}에 저장되었습니다.")
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")

if __name__ == "__main__":
    main()
