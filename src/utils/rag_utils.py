import requests
from bs4 import BeautifulSoup
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import VectorParams, Distance
from sentence_transformers import SentenceTransformer
from typing import Tuple, List, Dict, Any
import tiktoken
import os
import logging
import PyPDF2
import sys

MAX_TOKEN_COUNT_FOR_SOURCE_TEXT = 3000
logger = logging.getLogger(__name__)

# srcディレクトリをモジュール検索パスに追加
# これは、rag_utils.py が src/utils ディレクトリにある場合に必要です
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Qdrantクライアントを初期化
client = QdrantClient(host="localhost", port=6333)

# コレクションが存在しない場合のみ作成
if not client.collection_exists(collection_name="webpages"):
    client.create_collection(
        collection_name="webpages",
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )
    logger.info("コレクション 'webpages' を作成しました。")
else:
    logger.info("コレクション 'webpages' は既に存在します。")


def process_pdf(url: str, document_id: int):
    """
    PDFファイルのコンテンツを取得し、ベクトル化してQdrantに格納する
    """
    try:
        # PDFファイルをダウンロード
        response = requests.get(url)
        response.raise_for_status()
        with open("temp.pdf", "wb") as f:
            f.write(response.content)

        pdf_reader = PyPDF2.PdfReader(open("temp.pdf", "rb"))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        title = pdf_reader.metadata.get("/Title", "")

        # Sentence-BERTでテキストをベクトル化 (1536次元のモデルを使用)
        model = SentenceTransformer("gtr-t5-xl")
        vector = model.encode(text)

        # ベクトルをQdrantに格納
        client.upsert(
            collection_name="webpages",
            points=[
                models.PointStruct(
                    id=document_id,
                    vector=vector,
                    payload={"url": url, "title": title, "content": text},
                )
            ],
        )
        logger.info(
            f"PDFファイル '{title}' をQdrantに格納しました。 (ID: {document_id})"
        )

    except Exception as e:
        logger.error(f"PDFファイルの処理中にエラーが発生しました: {url}, エラー: {e}")


def query_index_use_user_question(user_question: str) -> List[Dict[str, Any]]:
    """
    貰った質問文に対して、Qdrantを使って関連するドキュメントを返すよ
    Args:
        user_question (str): ユーザーからの質問文
    Returns:
        results (List[Dict[str, Any]]): 関連するドキュメント
    """
    from src.modules.create_answer import create_embedding

    # Qdrantでベクトル検索
    search_result = client.search(
        collection_name="webpages",
        query_vector=create_embedding(user_question),
        limit=10,
    )

    # 検索結果をフォーマット
    results = []
    for point in search_result:
        results.append(
            {
                "content": point.payload.get("content", ""),
                "url": point.payload.get("url", ""),
                "title": point.payload.get("title", ""),
                "score": point.score,
            }
        )

    return results


def format_query_results(query_results: List[Dict[str, Any]]) -> str:
    """
    貰った関連するドキュメントを、
    ChatGPTのトークン数上限の限界まで繋げていき、あとはくっつけるだけの状態にするよ
    Args:
        query_results (List[Dict[str, Any]]): 関連するドキュメント
    Returns:
        source_text (str): １つにまとめた関連ドキュメント
    """
    source_text = ""
    for i, result in enumerate(query_results):
        subject = result.get("title", "")
        contents = result.get("content", "")  # 辞書にキーがない場合の処理を追加

        # トークン数チェック
        token_count = calc_token_count(
            model=os.environ.get("OPENAI_CHAT_COMPLETION_MODEL", None),
            text=source_text
            + f"[{i}]:"
            + f"{i} subject: {subject}, contents: {contents}"
            + "\n",
        )

        if token_count > MAX_TOKEN_COUNT_FOR_SOURCE_TEXT:
            break

        source_text += f"{i} subject: {subject}, contents: {contents}"

    return source_text


def calc_token_count(model: str, text: str) -> int:
    """
    貰ったテキストのトークン数を計算するよ
    Args:
        model (str): モデル名
        text (str): テキスト
    Returns:
        token_count (int): トークン数
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception:
        logger.error(
            f"トークンカウントの計算中にエラーが発生しました in calc_token_count",
            exc_info=True,
        )
        return 0


def create_response(message: Tuple[str, str]) -> str:
    """
    貰った質問文に対して、openaiのChatGPTを使って返答を生成するよ
    Args:
        message (Tuple[str,str]): ユーザーからの質問文
    Returns:
        response (str): ChatGPTの返答
    """
    from src.utils.chatbot_utils import init_openai

    openai_client = init_openai()
    try:
        response = openai_client.chat.completions.create(
            messages=message, model=os.environ["OPENAI_MODEL"]
        )
        message = response.choices[0].message
        return message.content
    except Exception:
        logger.error("Error occurred while getting chat response.", exc_info=True)
        return None


if __name__ == "__main__":
    # PDFファイルのURLとID
    pdf_url = "https://rais.skr.u-ryukyu.ac.jp/wordpress/wp-content/uploads/jikanwari/R6-2/01gakubu/09kokusaitiiki/kokuti_keizai_2.pdf"
    pdf_id = 2
    process_pdf(pdf_url, pdf_id)

    # ユーザーからの質問
    user_question = "琉球大学の経済学部について教えてください"

    # Qdrantを使って関連ドキュメントを検索
    query_results = query_index_use_user_question(user_question)

    # 検索結果を整形
    source_text = format_query_results(query_results)

    # ChatGPTに質問を送信
    message = [{"role": "user", "content": f"{source_text}\n{user_question}"}]
    answer = create_response(message)

    # 回答を表示
    print(f"質問: {user_question}")
    print(f"回答: {answer}")
