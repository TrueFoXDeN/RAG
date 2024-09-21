import pymupdf


def extract_from_pdf(file):
    doc = pymupdf.open("bucket_content/" + file)
    pages = []
    for page in doc:
        pages.append(page.get_text())

    return pages
