"""Build an index from a folder of PDFs.

Usage: python ingest.py <index_name> <folder_with_pdfs>
"""
import json
import sys
from pathlib import Path

import numpy as np

from rag import INDEX_DIR, chunk_text, embed_many, read_pdf_pages

name, folder = sys.argv[1], Path(sys.argv[2])
chunks = []
for pdf in sorted(folder.glob('*.pdf')):
    for page_number, text in read_pdf_pages(pdf):
        for piece in chunk_text(text):
            chunks.append({'source': pdf.name, 'page': page_number, 'text': piece})
print(f'{len(chunks)} chunks from {folder}; embedding...')

vectors = embed_many([c['text'] for c in chunks])

INDEX_DIR.mkdir(exist_ok=True)
np.save(INDEX_DIR / f'{name}.npy', np.array(vectors, dtype=np.float32))
(INDEX_DIR / f'{name}.json').write_text(json.dumps(chunks))
print(f'saved index "{name}"')
