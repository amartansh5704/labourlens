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
        temperature: float = 0.15,
        max_tokens: Optional[int] = None,
    ) -> str:
        try:
            # detect if this is a detailed request
            # give more tokens for detailed answers
            is_detailed = (
                "DETAILED" in prompt or
                "comprehensive" in prompt.lower() or
                "thoroughly" in prompt.lower() or
                "elaborate" in prompt.lower() or
                "in depth" in prompt.lower() or
                "step by step" in prompt.lower()
            )

            token_limit = (
                2048 if is_detailed
                else max_tokens or 800
            )

            logger.debug(
                f"Groq request: {len(prompt)} chars "
                f"| temp={temperature} "
                f"| tokens={token_limit} "
                f"| detailed={is_detailed}"
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
                max_tokens=token_limit,
                temperature=temperature,
                top_p=0.9,
                frequency_penalty=0.1,
            )

            answer = response.choices[0].message.content
            logger.debug(
                f"Groq response: {len(answer)} chars"
            )
            return answer.strip()

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