import os
from typing import Optional, Tuple

import gradio as gr
import pickle
from query_data import get_chain
from threading import Lock

with open("vectorstore.pkl", "rb") as f:
    vectorstore = pickle.load(f)


def set_hf_token(api_key: str):
    """Set the api key and return chain.
    If no api_key, then None is returned.
    """
    if api_key:
        os.environ["hf_token"] = api_key
        chain = get_chain(vectorstore)
        os.environ["hf_token"] = ""
        return chain

class ChatWrapper:

    def __init__(self):
        self.lock = Lock()
    def __call__(
        self, api_key: str, inp: str, history: Optional[Tuple[str, str]], chain
    ):
        """Execute the chat functionality."""
        self.lock.acquire()
        try:
            history = history or []
            # If chain is None, that is because no API key was provided.
            if chain is None:
                history.append((inp, "Please paste your HuggingFace token to use"))
                return history, history
            # Set OpenAI key
            # import openai
            # openai.api_key = api_key
            # Run chain and append input.
            output = chain({"question": inp, "chat_history": history})["answer"]
            history.append((inp, output))
        except Exception as e:
            raise e
        finally:
            self.lock.release()
        return history, history

chat = ChatWrapper()

block = gr.Blocks(css=".gradio-container {background-color: lightgray}")

with block:
    with gr.Row():
        gr.Markdown("<h3><center>Chat-Your-Data (State-of-the-Union)</center></h3>")

        hf_token_textbox = gr.Textbox(
            placeholder="Paste your HuggingFace Token (hf_...)",
            show_label=False,
            lines=1,
            type="password",
        )

    chatbot = gr.Chatbot()

    with gr.Row():
        message = gr.Textbox(
            label="What's your question?",
            placeholder="Ask questions about the most recent state of the union",
            lines=1,
        )
        submit = gr.Button(value="Send", variant="secondary").style(full_width=False)

    gr.Examples(
        examples=[
            "What did the president say about Kentaji Brown Jackson",
            "Did he mention Stephen Breyer?",
            "What was his stance on Ukraine",
        ],
        inputs=message,
    )

    gr.HTML("Demo application of a LangChain chain.")

    gr.HTML(
        "<center>Powered by <a href='https://github.com/hwchase17/langchain'>LangChain 🦜️🔗</a></center>"
    )

    state = gr.State()
    agent_state = gr.State()

    submit.click(chat, inputs=[hf_token_textbox, message, state, agent_state], outputs=[chatbot, state])
    message.submit(chat, inputs=[hf_token_textbox, message, state, agent_state], outputs=[chatbot, state])

    hf_token_textbox.change(
        set_hf_token,
        inputs=[hf_token_textbox],
        outputs=[agent_state],
    )

block.launch(debug=True)