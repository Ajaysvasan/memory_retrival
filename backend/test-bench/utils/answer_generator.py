"""Simple answer generator that extracts relevant information"""
import re
from typing import List


def generate_answer(query: str, documents: List[str], max_length: int = 500) -> str:
    """
    Generate a focused answer from documents based on the query
    
    Args:
        query: The input query
        documents: List of document texts
        max_length: Maximum length of the answer
        
    Returns:
        Focused answer string
    """
    if not documents:
        return "No relevant information found."
    
    try:
        # Extract query keywords (remove common stop words)
        query_lower = query.lower()
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'what', 'how', 'why', 'when', 'where', 'who', 'which'}
        query_words = set(re.findall(r'\b\w+\b', query_lower)) - stop_words
        
        if not query_words:
            query_words = set(re.findall(r'\b\w+\b', query_lower))
        
        # Score sentences by relevance
        all_sentences = []
        for doc in documents:
            if not doc or not doc.strip():
                continue
                
            # Split into sentences
            sentences = re.split(r'[.!?]+', doc)
            for sent in sentences:
                sent = sent.strip()
                if len(sent) < 15:  # Skip very short sentences
                    continue
                
                # Score sentence by query word matches
                sent_lower = sent.lower()
                sent_words = set(re.findall(r'\b\w+\b', sent_lower))
                matches = len(query_words.intersection(sent_words))
                score = matches / len(query_words) if query_words else 0.1
                
                all_sentences.append((score, sent))
        
        # Sort by relevance
        all_sentences.sort(key=lambda x: x[0], reverse=True)
        
        # Build answer from top sentences
        answer_parts = []
        current_length = 0
        
        for score, sentence in all_sentences:
            if score >= 0 and current_length + len(sentence) < max_length:
                answer_parts.append(sentence)
                current_length += len(sentence)
            elif current_length > 100:  # Have at least some content
                break
        
        if not answer_parts:
            # Fallback: use first few sentences from top document
            top_doc = documents[0] if documents else ""
            if top_doc:
                sentences = re.split(r'[.!?]+', top_doc)
                answer_parts = [s.strip() for s in sentences[:2] if len(s.strip()) > 15]
        
        if not answer_parts:
            # Last resort: use first part of first document
            if documents and documents[0]:
                first_doc = documents[0]
                # Take first reasonable chunk
                answer_parts = [first_doc[:max_length].strip()]
        
        answer = ". ".join(answer_parts)
        
        # Clean up and ensure it ends properly
        answer = answer.strip()
        if answer and not answer[-1] in '.!?':
            answer += "."
        
        # Truncate if too long
        if len(answer) > max_length:
            answer = answer[:max_length].rsplit('.', 1)[0] + "."
        
        return answer if answer else "No relevant information found."
        
    except Exception as e:
        # Fallback: return first document chunk
        if documents and documents[0]:
            return documents[0][:max_length] + ("..." if len(documents[0]) > max_length else "")
        return "No relevant information found."
