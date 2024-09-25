from src.utils.rag_utils import query_index_use_user_question, format_query_results


def add_sentense(question: str) -> str:
    """
    貰った質問文に対して、関連するドキュメントを検索し、
    質問文と関連ドキュメントを1つにまとめることで、ChatGPTに聞く質問文を作るよ
    """
    query_results = query_index_use_user_question(question)
    source_text = format_query_results(query_results)
    question_with_source = f"[質問]:" + source_text + "\n[参考文献]:" + question
    return question_with_source
