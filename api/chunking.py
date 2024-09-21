
import pymupdf
from semantic_chunkers import StatisticalChunker
from semantic_router.encoders import HuggingFaceEncoder

encoder = HuggingFaceEncoder()


def load_pdf(path):
    doc = pymupdf.open(path)
    pages = []
    for page in doc:
        pages.append(page.get_text())
    return pages


def statistical_chunking(pages):
    chunker = StatisticalChunker(encoder=encoder)
    chunks = chunker(docs=pages)
    return chunks


# def consecutive_chunking(pages):
#     chunker = ConsecutiveChunker(encoder=encoder, score_threshold=0.1)
#     chunks = chunker(docs=pages)
#     for chunk in chunks:
#         print(chunk)
#     return chunks
#
#
# def cumulative_chunking(pages):
#     chunker = CumulativeChunker(encoder=encoder, score_threshold=0.3)
#     chunks = chunker(docs=pages)
#     for chunk in chunks:
#         print(chunk)
#     return chunks


def sanitize_chunks(chunks):
    sanitized_chunks = []
    for page_chunk in chunks:
        sanitized_page_chunk = []
        for chunk in page_chunk:
            merged_text = " ".join(chunk.splits)
            sanitized_page_chunk.append(merged_text)
        sanitized_chunks.append(sanitized_page_chunk)
    # print(sanitized_chunks)
    return sanitized_chunks


def split_chunks(pages):
    chunks = statistical_chunking(pages)
    sanatized_chunks = sanitize_chunks(chunks)
    return sanatized_chunks
