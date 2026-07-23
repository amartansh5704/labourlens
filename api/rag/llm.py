# api/rag/llm.py
from groq import Groq
from api.core.config import settings
from api.rag.prompts import SYSTEM_PROMPT
from loguru import logger
from typing import Optional


class GroqLLM:

    def __init__(self):
        if not settings.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found")

        self.client = Groq(api_key=settings.GROQ_API_KEY)
        self.model = settings.GROQ_MODEL
        self.max_tokens = settings.MAX_ANSWER_TOKENS
        logger.info(f"GroqLLM ready: {self.model}")

    def generate(
        self,
        prompt: str,
        temperature: float = 0.30,
        max_tokens: Optional[int] = None,
    ) -> str:
        try:
            logger.debug(
                f"Groq request: {len(prompt)} chars "
                f"temp={temperature}"
            )

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=max_tokens or 512,
                temperature=temperature,
                top_p=0.9,          # reduce randomness
                frequency_penalty=0.3,
            )

            answer = response.choices[0].message.content
            logger.debug(
                f"Groq response: {len(answer)} chars"
            )
            return answer

        except Exception as e:
            logger.error(f"Groq error: {e}")
            raise

    def test_connection(self) -> bool:
        try:
            response = self.generate(
                "Say exactly: CONNECTION_OK",
                max_tokens=10
            )
            return "CONNECTION_OK" in response
        except Exception:
            return False