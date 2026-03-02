import os
import time
from typing import Callable


def create_judge_agent(
    model_name: str = "llama3.2:3b",
    api_key: str | None = None,
) -> Callable[[str, str, str], str]:
    """
    Create a simple LLM-as-judge function.
    - api_key=None  → uses local Ollama (for run_chunking_experiment.py)
    - api_key="xxx" → uses Gemini     (for run_eval.py)
    """

    if api_key is None:
        from langchain_ollama import ChatOllama
        llm = ChatOllama(
            model=model_name,
            temperature=0.0,
            timeout=300,
        )
    else:
        from langchain_google_genai import ChatGoogleGenerativeAI
        model_name = os.getenv("GEMINI_MODEL_JUDGE", model_name)
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.0,
            convert_system_message_to_human=True,
        )

    def judge(question: str, gold_answer: str, predicted_answer: str) -> str:
        prompt = f"""You are an automatic evaluator. Judge fairly and accurately.

        Question: {question}

        Gold Answer: {gold_answer}

        Predicted Answer: {predicted_answer}

        Rules:
        - CORRECT: predicted answer conveys the same main idea as the gold answer, even with different wording
        - CORRECT: predicted answer is partially correct and covers the key point of the question
        - INCORRECT: predicted answer says "I don't have enough information" or similar refusal
        - INCORRECT: predicted answer is about a completely different topic
        - INCORRECT: predicted answer contradicts the gold answer

        Reply with exactly one word: CORRECT or INCORRECT"""

        if api_key is not None:
            # Gemini — retry on rate limit
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
        else:
            # Ollama — simple call
            try:
                resp = llm.invoke(prompt)
            except Exception as e:
                print(f"  ERROR in Judge: {e}")
                return "INCORRECT"

        text = getattr(resp, "content", str(resp))
        t_upper = text.upper().strip()

        has_correct   = "CORRECT"   in t_upper
        has_incorrect = "INCORRECT" in t_upper

        if has_correct and not has_incorrect:
            return "CORRECT"
        if has_incorrect and not has_correct:
            return "INCORRECT"
        if has_correct and has_incorrect:
            if t_upper.index("CORRECT") < t_upper.index("INCORRECT"):
                return "CORRECT"
            return "INCORRECT"

        # Fallback for small models that output yes/no
        if "YES" in t_upper:
            return "CORRECT"
        if "NO" in t_upper:
            return "INCORRECT"

        return "INCORRECT"

    return judge

#----------------------------------------------------------------------------------------------*---------
# import os
# import time
# from typing import Callable
#
# from langchain_google_genai import ChatGoogleGenerativeAI
#
#
# def create_judge_agent(
#     # Default to a free-tier friendly model.
#     # Override with env var `GEMINI_MODEL_JUDGE` or argument.
#     model_name: str = "gemini-2.5-flash",
#     api_key: str | None = None,
# ) -> Callable[[str, str, str], str]:
#     """
#     Create a simple LLM-as-judge function.
#
#     Returns a callable:
#         judge(question, gold_answer, predicted_answer) -> 'CORRECT' | 'INCORRECT'
#     """
#     if api_key is None:
#         api_key = os.getenv("GOOGLE_API_KEY")
#     if not api_key:
#         raise ValueError("Please set GOOGLE_API_KEY env var or pass api_key explicitly.")
#
#     model_name = os.getenv("GEMINI_MODEL_JUDGE", model_name)
#     llm = ChatGoogleGenerativeAI(
#         model=model_name,
#         google_api_key=api_key,
#         temperature=0.0,
#         convert_system_message_to_human=True,
#     )
#
#     # def judge(question: str, gold_answer: str, predicted_answer: str) -> str:
#     #     prompt = f"""
#     #     You are an automatic evaluator.
#     #
#     #     Question: {question}
#     #
#     #     Gold Answer: {gold_answer}
#     #
#     #     Predicted Answer: {predicted_answer}
#     #
#     #     If the predicted answer correctly answers the question (allowing minor wording differences),
#     #     respond with exactly one word: CORRECT
#     #     Otherwise respond with exactly one word: INCORRECT
#     #     Do NOT add any explanation or extra words.
#     #     """
#     #     # Basic backoff for 429/rate limits.
#     #     resp = None
#     #     for attempt in range(5):
#     #         try:
#     #             resp = llm.invoke(prompt)
#     #             break
#     #         except Exception as e:
#     #             msg = str(e)
#     #             if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
#     #                 time.sleep(2 ** attempt)
#     #                 continue
#     #             raise
#     #     text = getattr(resp, "content", str(resp))
#     #     t_upper = text.upper()
#     #
#     #     # Robust parsing: look for CORRECT / INCORRECT anywhere in the output.
#     #     has_correct = "CORRECT" in t_upper
#     #     has_incorrect = "INCORRECT" in t_upper
#     #
#     #     if has_correct and not has_incorrect:
#     #         return "CORRECT"
#     #     if has_incorrect and not has_correct:
#     #         return "INCORRECT"
#     #     if has_correct and has_incorrect:
#     #         # If both appear, fall back to the first occurrence.
#     #         if t_upper.index("CORRECT") < t_upper.index("INCORRECT"):
#     #             return "CORRECT"
#     #         return "INCORRECT"
#     #
#     #     # Fallback when the model completely ignores instructions.
#     #     return "INCORRECT"
#     def judge(question: str, gold_answer: str, predicted_answer: str) -> str:
#         prompt = f"""You are an automatic evaluator. Be LENIENT in your judgment.
#
#         Question: {question}
#
#         Gold Answer: {gold_answer}
#
#         Predicted Answer: {predicted_answer}
#
#         Evaluation rules:
#         - If the predicted answer captures the MAIN IDEA or KEY CONCEPTS of the gold answer, respond CORRECT
#         - Minor missing details, different wording, or partial answers are still CORRECT
#         - Only respond INCORRECT if the predicted answer is completely wrong, contradicts the gold answer, or says "I don't have enough information"
#         - Allow different phrasing, synonyms, and partial information
#
#         Reply with exactly one word: CORRECT or INCORRECT"""
#
#         resp = None
#         for attempt in range(5):
#             try:
#                 resp = llm.invoke(prompt)
#                 break
#             except Exception as e:
#                 msg = str(e)
#                 if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
#                     time.sleep(2 ** attempt)
#                     continue
#                 raise
#
#         text = getattr(resp, "content", str(resp))
#         t_upper = text.upper()
#
#         has_correct = "CORRECT" in t_upper
#         has_incorrect = "INCORRECT" in t_upper
#
#         if has_correct and not has_incorrect:
#             return "CORRECT"
#         if has_incorrect and not has_correct:
#             return "INCORRECT"
#         if has_correct and has_incorrect:
#             if t_upper.index("CORRECT") < t_upper.index("INCORRECT"):
#                 return "CORRECT"
#             return "INCORRECT"
#
#         return "INCORRECT"
#
#     return judge



