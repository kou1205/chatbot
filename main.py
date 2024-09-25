import logging
from typing import List, Tuple
from src.modules.add_sentence import add_sentense
import gradio as gr
from src.modules.create_answer import create_answer

logger = logging.getLogger(__name__)


def handle_submit(
    user_message: str, chat_history: List[Tuple[str, str]]
) -> Tuple[str, List[Tuple[str, str]]]:
    """
    gradioのchatbot機能を使うために必要な関数だよ
    質問文とそれに対する回答のリストを返すよ
    """
    bot_message = create_answer(add_sentense(user_message))
    chat_history.append((user_message, bot_message))
    return "", chat_history


with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])

    msg.submit(handle_submit, [msg, chatbot], [msg, chatbot])


if __name__ == "__main__":
    demo.launch()
