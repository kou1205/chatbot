from src.utils.rag_utils import create_response
from typing import Optional, List
import logging
import os

logger = logging.getLogger(__name__)


def create_answer(question: str) -> str:
    """
    貰った質問文に対して、ChatGPTを使って返答を生成するよ
    Args:
        question (str): ユーザーからの質問文
    Returns:
        bot_message (str): ChatGPTの返答
    """
    from src.utils.chatbot_utils import format_user_message

    message = format_user_message(question)
    bot_message = create_response(message)
    return bot_message


def create_embedding(text: str) -> Optional[List[float]]:
    """
    貰った質問文に対して、openai機能を使って、ベクトル化したものを返すよ
    Args:
        text (str): ユーザーからの質問文
    Returns:
        embedding (List[float]): ベクトル化された質問文
    """
    from src.utils.chatbot_utils import init_openai

    openai_client = init_openai()
    try:
        response = (
            openai_client.embeddings.create(
                input=[text],
                model=os.environ.get("OPENAI_EMBEDDING_MODEL", None),
            )
            .data[0]
            .embedding
        )
        return response
    except Exception:
        logger.error(
            "embeddingの作成中にエラーが発生しました in create_embedding", exc_info=True
        )
        return None
