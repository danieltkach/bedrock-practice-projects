import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3
import numpy as np
from pypdf import PdfReader

REGION = 'us-east-1'
EMBED_MODEL = 'amazon.titan-embed-text-v2:0'
EMBED_DIMS = 512
INDEX_DIR = Path(__file__).parent / 'indexes'

runtime = boto3.client('bedrock-runtime', region_name=REGION)


def read_pdf_pages(path):
    """Return [(page_number, text), ...] for a PDF. Page numbers start at 1."""
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = re.sub(r'\s+', ' ', page.extract_text() or '').strip()
        if text:
            pages.append((i, text))
    return pages


def chunk_text(text, size=1200, overlap=200):
    """Split text into overlapping pieces, cutting on a space where possible."""
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = text.rfind(' ', start + size // 2, end)
            end = cut if cut != -1 else end
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def embed(text):
    """Turn text into a vector of numbers whose direction captures its meaning."""
    response = runtime.invoke_model(
        modelId=EMBED_MODEL,
        body=json.dumps({'inputText': text, 'dimensions': EMBED_DIMS, 'normalize': True}),
    )
    return json.loads(response['body'].read())['embedding']


def embed_many(texts, workers=8):
    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(embed, texts))


class Index:
    """One searchable collection: a matrix of vectors plus the chunk metadata."""

    def __init__(self, name):
        self.name = name
        self.vectors = np.load(INDEX_DIR / f'{name}.npy')
        self.chunks = json.loads((INDEX_DIR / f'{name}.json').read_text())

    def search(self, query_vector, k=5):
        # vectors are normalized, so a dot product is the cosine similarity
        scores = self.vectors @ np.array(query_vector)
        best = np.argsort(scores)[::-1][:k]
        return [{**self.chunks[i], 'score': float(scores[i]), 'index': self.name} for i in best]


def available_indexes():
    return sorted(p.stem for p in INDEX_DIR.glob('*.npy'))
