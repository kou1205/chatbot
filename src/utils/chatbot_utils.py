import logging
import os
from typing import Tuple
from openai import OpenAI

logger = logging.getLogger(__name__)


def format_user_message(question: str):
    """
    貰った質問文をChatGPTに聞く時に、リクエストの形式に合わせて整形するよ
    Args:
        question (str): ユーザーからの質問文
    Returns:
        messages (List[Dict[str, str]]): ChatGPTに聞く時のリクエストの形式
    """
    messages = [
        {
            "role": "system",
            "content": "あなたは授業文書から適切な情報を抽出し、学生の質問に答えるアシスタントです。",
        },
        {"role": "user", "content": question},
    ]
    return messages


def init_openai():
    """
    openaiの機能を使うためのクライアントを作成するよ
    Returns:
        openai_client (OpenAI): openaiのクライアント
    """
    assert (
        "OPENAI_EMBEDDING_MODEL" in os.environ
    ), "OPENAI_EMBEDDING_MODEL environment variable is not set"
    assert (
        "OPENAI_CHAT_COMPLETION_MODEL" in os.environ
    ), "OPENAI_CHAT_COMPLETION_MODEL environment variable is not set"
    openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return openai_client
