"""Conversational legal assistant over one or more indexes.

Usage: python assistant.py
Commands: /indexes  /use <name|all>  /summary <part of a filename>  /quit
"""
from pathlib import Path

from rag import Index, available_indexes, chunk_text, embed, read_pdf_pages, runtime

ANSWER_MODEL = 'us.anthropic.claude-sonnet-4-5-20250929-v1:0'
FAST_MODEL = 'us.anthropic.claude-haiku-4-5-20251001-v1:0'
DOCS_DIR = Path(__file__).parent / 'data'
HISTORY_TURNS = 6  # how many past messages the model sees

SYSTEM = (
    'You are a legal research assistant. Answer ONLY from the numbered sources provided. '
    'Cite every claim with its source number in brackets, like [1] or [2][3]. '
    'If the sources do not contain the answer, say so plainly instead of guessing. '
    'Never infer facts the sources do not state (for example, do not label an opinion "per curiam" '
    'or name an author unless a source says so); say the sources do not show it. '
    'You provide research help, not legal advice.'
)


def ask(model, system, messages, max_tokens=1000):
    response = runtime.converse(
        modelId=model,
        system=[{'text': system}],
        messages=[{'role': m['role'], 'content': [{'text': m['content']}]} for m in messages],
        inferenceConfig={'maxTokens': max_tokens},
    )
    blocks = response['output']['message']['content']
    return ''.join(b['text'] for b in blocks if 'text' in b)


def standalone_question(history, question):
    """Rewrite a follow-up ('what about the dissent?') into a question that makes sense alone."""
    if not history:
        return question
    transcript = '\n'.join(f"{m['role']}: {m['content']}" for m in history[-HISTORY_TURNS:])
    return ask(
        FAST_MODEL,
        'Rewrite the final user question so it is fully self-contained, using the conversation for context. '
        'Return only the rewritten question.',
        [{'role': 'user', 'content': f'{transcript}\n\nFinal question: {question}'}],
        max_tokens=200,
    ).strip()


def retrieve(indexes, query, k=5):
    vector = embed(query)
    hits = [hit for index in indexes for hit in index.search(vector, k)]
    return sorted(hits, key=lambda h: h['score'], reverse=True)[:k]


def answer(indexes, history, question):
    query = standalone_question(history, question)
    hits = retrieve(indexes, query)
    sources = '\n\n'.join(
        f"[{i}] ({h['source']}, page {h['page']})\n{h['text']}" for i, h in enumerate(hits, start=1)
    )
    prompt = f'Sources:\n{sources}\n\nQuestion: {question}'
    messages = history[-HISTORY_TURNS:] + [{'role': 'user', 'content': prompt}]
    text = ask(ANSWER_MODEL, SYSTEM, messages)
    citations = '\n'.join(f"  [{i}] {h['source']}, p. {h['page']} ({h['index']})" for i, h in enumerate(hits, start=1))
    return text, citations


def summarize(part):
    """Map-reduce summary: summarize each section, then combine into an executive summary."""
    matches = [p for p in DOCS_DIR.rglob('*.pdf') if part.lower() in p.name.lower()]
    if not matches:
        return f'No document matches "{part}".'
    pdf = matches[0]
    full_text = ' '.join(text for _, text in read_pdf_pages(pdf))
    sections = chunk_text(full_text, size=20000, overlap=0)
    notes = [
        ask(FAST_MODEL, 'Summarize this part of a legal document in 150 words, keeping holdings, parties and reasoning.',
            [{'role': 'user', 'content': s}], max_tokens=400)
        for s in sections
    ]
    final = ask(
        ANSWER_MODEL,
        'Write an executive summary for a busy lawyer: parties, question presented, holding, reasoning, '
        'and any dissent. Under 250 words.',
        [{'role': 'user', 'content': '\n\n'.join(notes)}],
        max_tokens=600,
    )
    return f'Summary of {pdf.name}:\n{final}'


def main():
    names = available_indexes()
    if not names:
        raise SystemExit('No indexes found. Run ingest.py first.')
    loaded = {n: Index(n) for n in names}
    active = list(loaded.values())
    history = []
    print(f'Loaded indexes: {", ".join(names)}. Type a question, or /quit.')

    while True:
        line = input('\nyou> ').strip()
        if not line:
            continue
        if line == '/quit':
            break
        if line == '/indexes':
            print('Available:', ', '.join(names), '| active:', ', '.join(i.name for i in active))
        elif line.startswith('/use '):
            choice = line[5:].strip()
            active = list(loaded.values()) if choice == 'all' else [loaded[choice]] if choice in loaded else active
            print('Active:', ', '.join(i.name for i in active))
        elif line.startswith('/summary '):
            print(summarize(line[9:].strip()))
        else:
            text, citations = answer(active, history, line)
            print(f'\n{text}\n\nSources:\n{citations}')
            history += [{'role': 'user', 'content': line}, {'role': 'assistant', 'content': text}]


if __name__ == '__main__':
    main()
