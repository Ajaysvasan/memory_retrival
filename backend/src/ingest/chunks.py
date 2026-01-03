def split_into_blocks(text: str):
    blocks = text.split("\n\n")
    return [b.strip() for b in blocks if b.strip()]


def assemble_chunks(blocks, max_words=350, overlap_words=50):
    chunks = []
    current = []
    current_len = 0

    for block in blocks:
        words = block.split()
        block_len = len(words)

        if block_len > max_words:
            if current:
                chunks.append(" ".join(current))
                current = []
                current_len = 0

            for i in range(0, block_len, max_words):
                sub = " ".join(words[i : i + max_words])
                chunks.append(sub)
            continue

        if current_len + block_len > max_words:
            chunks.append(" ".join(current))

            if overlap_words > 0:
                overlap = " ".join(" ".join(current).split()[-overlap_words:])
                current = [overlap]
                current_len = len(overlap.split())
            else:
                current = []
                current_len = 0
        current.append(block)
        current_len += block_len

    if current:
        chunks.append(" ".join(current))

    return chunks


def process_document(document):
    chunks = chunk_document(document)
    chunks = deduplicate_chunks(chunks)
    return chunks


def deduplicate_chunks(chunks):
    seen = set()
    unique = []

    for c in chunks:
        h = hash(c["text"])
        if h not in seen:
            seen.add(h)
            unique.append(c)

    return unique


def chunk_document(document, max_words=350, overlap_words=50):
    blocks = split_into_blocks(document["text"])
    chunks = assemble_chunks(blocks, max_words, overlap_words)

    return [
        {
            "text": chunk,
            "source": document["source"],
            "chunk_index": idx,
            "word_count": len(chunk.split()),
        }
        for idx, chunk in enumerate(chunks)
    ]
