"""Web front end for the legal assistant, built with Gradio.

Usage: python app.py   (then open http://127.0.0.1:7860)
Reuses the same retrieval, answering and summary code as the command-line assistant.
"""
import gradio as gr

from assistant import DOCS_DIR, answer, summarize
from rag import Index, available_indexes

INTRO = (
    '# Legal Assistant\n'
    'Ask questions about U.S. Supreme Court opinions and oral arguments. Answers use only the retrieved '
    'passages and cite file and page. **Research aid only, not legal advice: check every answer against the cited page.**'
)


def load_indexes():
    names = available_indexes()
    if not names:
        raise SystemExit('No indexes found. Run ./run.sh ingest first.')
    return {name: Index(name) for name in names}


def opinion_files():
    return sorted(p.name for p in (DOCS_DIR / 'opinions').glob('*.pdf'))


def message_text(content):
    """Gradio 6 returns message content as a list of {'text', 'type'} blocks; older versions use a string."""
    if isinstance(content, str):
        return content
    return ''.join(block.get('text', '') for block in content if isinstance(block, dict))


def make_app(indexes):
    def respond(message, history, selected):
        """One chat turn: returns (cleared textbox, updated chat, sources panel)."""
        message = (message or '').strip()
        if not message:
            return '', history, ''
        active = [indexes[name] for name in selected] or list(indexes.values())
        # Gradio stores extra fields per message; the model only needs role and text.
        past = [{'role': m['role'], 'content': message_text(m['content'])} for m in history]
        try:
            text, citations = answer(active, past, message)
        except Exception as error:  # most often expired AWS credentials or a missing model access form
            text, citations = f'Sorry, the request failed: {error}', ''
        history = history + [{'role': 'user', 'content': message}, {'role': 'assistant', 'content': text}]
        sources = f'**Sources for the last answer**\n```\n{citations}\n```' if citations else ''
        return '', history, sources

    def summarize_file(file_name):
        if not file_name:
            return 'Choose an opinion first.'
        try:
            return summarize(file_name)
        except Exception as error:
            return f'Sorry, the summary failed: {error}'

    with gr.Blocks(title='Legal Assistant') as demo:
        gr.Markdown(INTRO)
        with gr.Tab('Ask questions'):
            selected = gr.CheckboxGroup(list(indexes), value=list(indexes), label='Search in')
            chatbot = gr.Chatbot(height=420, label='Conversation')
            sources = gr.Markdown()
            box = gr.Textbox(placeholder='Ask a question, then press Enter', show_label=False)
            clear = gr.Button('Clear conversation')
            box.submit(respond, [box, chatbot, selected], [box, chatbot, sources], api_name='respond')
            clear.click(lambda: ([], ''), None, [chatbot, sources], api_name='clear')
        with gr.Tab('Executive summary'):
            gr.Markdown('Pick an opinion to get a short executive summary (takes about 20 to 60 seconds).')
            file_name = gr.Dropdown(opinion_files(), label='Opinion')
            button = gr.Button('Summarize', variant='primary')
            summary = gr.Markdown()
            button.click(summarize_file, file_name, summary, api_name='summarize')
    return demo


if __name__ == '__main__':
    make_app(load_indexes()).launch()
