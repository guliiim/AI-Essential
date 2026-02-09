import os
import time
from typing import Callable

from langchain_google_genai import ChatGoogleGenerativeAI


def create_judge_agent(
    # Default to a free-tier friendly model.
    # Override with env var `GEMINI_MODEL_JUDGE` or argument.
    model_name: str = "gemini-2.5-flash",
    api_key: str | None = None,
) -> Callable[[str, str, str], str]:
    """
    Create a simple LLM-as-judge function.

    Returns a callable:
        judge(question, gold_answer, predicted_answer) -> 'CORRECT' | 'INCORRECT'
    """
    if api_key is None:
        api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Please set GOOGLE_API_KEY env var or pass api_key explicitly.")

    model_name = os.getenv("GEMINI_MODEL_JUDGE", model_name)
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.0,
        convert_system_message_to_human=True,
    )

    def judge(question: str, gold_answer: str, predicted_answer: str) -> str:
        prompt = f"""
You are an automatic evaluator.

Question: {question}

Gold Answer: {gold_answer}

Predicted Answer: {predicted_answer}

If the predicted answer correctly answers the question (allowing minor wording differences),
respond with exactly one word: CORRECT
Otherwise respond with exactly one word: INCORRECT
Do NOT add any explanation or extra words.
"""
        # Basic backoff for 429/rate limits.
        resp = None
        for attempt in range(5):
            try:
                resp = llm.invoke(prompt)
                break
            except Exception as e:
                msg = str(e)
                if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
                    time.sleep(2 ** attempt)
                    continue
                raise
        text = getattr(resp, "content", str(resp))
        t_upper = text.upper()

        # Robust parsing: look for CORRECT / INCORRECT anywhere in the output.
        has_correct = "CORRECT" in t_upper
        has_incorrect = "INCORRECT" in t_upper

        if has_correct and not has_incorrect:
            return "CORRECT"
        if has_incorrect and not has_correct:
            return "INCORRECT"
        if has_correct and has_incorrect:
            # If both appear, fall back to the first occurrence.
            if t_upper.index("CORRECT") < t_upper.index("INCORRECT"):
                return "CORRECT"
            return "INCORRECT"

        # Fallback when the model completely ignores instructions.
        return "INCORRECT"

    return judge



