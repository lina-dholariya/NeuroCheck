import json
import os
import re
import pandas as pd
import requests
from typing import List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import random


class SimpleLLMService:
    def __init__(self,
                 intents_path: str = 'intents.json',
                 faq_csv_path: str = 'Mental_Health_FAQ.csv',
                 use_api: bool = True,
                 api_provider: str = 'openrouter'):
        self.intents_path = intents_path
        self.faq_csv_path = faq_csv_path
        self.use_api = use_api
        self.api_provider = api_provider

        self.intents = []
        self.intent_responses = {}

        self.faq_df = None
        self.vectorizer = None
        self.faq_tfidf = None

        # API Configuration
        self.huggingface_token = os.getenv('HUGGINGFACE_TOKEN', '')
        self.openai_api_key = os.getenv('OPENAI_API_KEY', '')
        self.openrouter_api_key = os.getenv('OPENROUTER_API_KEY', '')
        self.openrouter_model = os.getenv('OPENROUTER_MODEL', 'gpt-4o')

        self._load_intents()
        self._load_faq()

    def _load_intents(self) -> None:
        if not os.path.exists(self.intents_path):
            return
        with open(self.intents_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        intents = data.get('intents', [])
        for it in intents:
            tag = it.get('tag')
            patterns = it.get('patterns', [])
            responses = it.get('responses', [])
            if not tag:
                continue
            for p in patterns:
                if isinstance(p, str) and p.strip():
                    self.intents.append((tag, p.strip().lower()))
            if responses:
                self.intent_responses[tag] = responses

        # Ensure default greetings intent exists
        if not any(tag == "greeting" for tag, _ in self.intents):
            self.intents.append(("greeting", "hi"))
            self.intent_responses["greeting"] = ["Hello! How are you feeling today?", 
                                                 "Hi there! How’s your mood today?"]

    def _load_faq(self) -> None:
        if not os.path.exists(self.faq_csv_path):
            return
        encodings = ['utf-8', 'latin-1']
        for enc in encodings:
            try:
                df = pd.read_csv(self.faq_csv_path, encoding=enc)
                self.faq_df = df
                break
            except Exception:
                continue
        if self.faq_df is None:
            return

        cols = {c.lower(): c for c in self.faq_df.columns}
        q_col = cols.get('question') or cols.get('questions') or list(self.faq_df.columns)[0]
        a_col = cols.get('answer') or cols.get('answers') or list(self.faq_df.columns)[1]
        self.faq_df = self.faq_df.rename(columns={q_col: 'question', a_col: 'answer'})
        self.faq_df['question'] = self.faq_df['question'].astype(str)
        self.faq_df['answer'] = self.faq_df['answer'].astype(str)

        self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        self.faq_tfidf = self.vectorizer.fit_transform(self.faq_df['question'])

    @staticmethod
    def _normalize(text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)
        return text

    def _match_intent(self, message: str) -> Optional[str]:
        msg = self._normalize(message)
        for tag, pat in self.intents:
            if pat == msg or pat in msg:
                return tag
        return None

    def _intent_response(self, tag: str) -> Optional[str]:
        options = self.intent_responses.get(tag)
        if not options:
            return None
        return random.choice(options)

    def _retrieve_faq(self, message: str) -> Optional[Tuple[str, str, float]]:
        if self.faq_df is None or self.vectorizer is None or self.faq_tfidf is None:
            return None
        query_vec = self.vectorizer.transform([message])
        cosine_similarities = linear_kernel(query_vec, self.faq_tfidf).flatten()
        idx = int(cosine_similarities.argmax())
        score = float(cosine_similarities[idx])
        if score < 0.3:  # Raise threshold to avoid loose matches
            return None
        question = self.faq_df.iloc[idx]['question']
        answer = self.faq_df.iloc[idx]['answer']

        # Truncate long answers for conversational tone
        max_length = 300  # characters
        if len(answer) > max_length:
            answer = answer[:max_length].rsplit('.', 1)[0] + '...'

        return question, answer, score

    def _call_openrouter_api(self, message: str, history: Optional[List[dict]] = None, model: Optional[str] = None) -> str:
        if not self.openrouter_api_key:
            return "OpenRouter API key not configured. Please set OPENROUTER_API_KEY."

        chosen_model = model or self.openrouter_model or 'gpt-4o'
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "HTTP-Referer": os.getenv('OPENROUTER_SITE_REF', 'https://neurocheck.app'),
            "X-Title": os.getenv('OPENROUTER_APP_TITLE', 'NeuroCheck Chatbot')
        }

        messages = [
            {"role": "system", "content": "You are a supportive and empathetic mental health assistant. Be concise, caring, and non-judgmental. Encourage professional help for serious concerns."}
        ]

        if history:
            for turn in history:
                if isinstance(turn, dict) and turn.get('role') in ('user', 'assistant') and isinstance(turn.get('content'), str):
                    messages.append({"role": turn['role'], "content": turn['content']})

        messages.append({"role": "user", "content": message})

        payload = {
            "model": chosen_model,
            "messages": messages,
            "max_tokens": 200,
            "temperature": 0.7
        }

        try:
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=15
            )
            if resp.status_code == 200:
                data = resp.json()
                return data['choices'][0]['message']['content'].strip()
            return "I'm here to listen. Could you tell me more about how you're feeling?"
        except Exception as e:
            print(f"OpenRouter API error: {e}")
            return "I'm here to support you. Please share what's on your mind."

    # Placeholder stubs for other APIs (implement as needed)
    def _call_huggingface_api(self, message: str) -> str:
        return ""

    def _call_free_api(self, message: str) -> str:
        return ""

    def _call_openai_api(self, message: str) -> str:
        return ""

    def generate_response(self, message: str, chat_history: Optional[List[dict]] = None) -> dict:
        """Generate a response for the given user message with priority:
        1) Intent
        2) FAQ
        3) OpenRouter API
        4) Hugging Face API
        5) Free API
        6) Empathetic fallback
        """
        if not message or not message.strip():
            return {
                'source': 'intent',
                'response': 'Please share what is on your mind. I am here to listen.',
                'confidence': 0.0
            }

        # 1) Intent
        tag = self._match_intent(message)
        if tag:
            resp = self._intent_response(tag)
            if resp:
                return {
                    'source': 'intent',
                    'intent': tag,
                    'response': resp,
                    'confidence': 0.9
                }

        # 2) FAQ
        retrieved = self._retrieve_faq(message)
        if retrieved:
            question, answer, score = retrieved
            return {
                'source': 'faq',
                'matched_question': question,
                'response': answer,
                'confidence': score
            }

        # 3) External LLMs
        if self.use_api:
            # OpenRouter primary
            if self.openrouter_api_key:
                api_response = self._call_openrouter_api(message, history=chat_history)
                if api_response and len(api_response.strip()) > 0:
                    return {
                        'source': 'api',
                        'provider': 'openrouter',
                        'response': api_response.strip(),
                        'confidence': 0.85
                    }

            # Hugging Face fallback
            if self.huggingface_token:
                api_response = self._call_huggingface_api(message)
                if api_response and len(api_response.strip()) > 0:
                    return {
                        'source': 'api',
                        'provider': 'huggingface',
                        'response': api_response.strip(),
                        'confidence': 0.8
                    }

            # Free API fallback
            api_response = self._call_free_api(message)
            if api_response and len(api_response.strip()) > 0:
                return {
                    'source': 'api',
                    'provider': 'free',
                    'response': api_response.strip(),
                    'confidence': 0.7
                }

        # 4) Empathetic fallback
        fallback_responses = [
            "I'm here for you. Could you share a bit more about how you're feeling?",
            "That sounds challenging. I'm listening if you want to talk about it.",
            "I understand this is difficult. What's on your mind right now?",
            "I'm here to support you. What would you like to discuss?",
            "That must be hard to deal with. I'm here to listen.",
            "I care about how you're feeling. What's troubling you today?",
            "You're not alone in this."
        ]
        return {
            'source': 'fallback',
            'response': random.choice(fallback_responses),
            'confidence': 0.5
        }


# Instantiate a single service instance
llm_service = SimpleLLMService()
