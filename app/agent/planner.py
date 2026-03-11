from typing import Dict, Any

def classify_query(query: str) -> str:
    """
    Classify the query using simple heuristic rules.
    Returns one of: "keyword", "conceptual", "vague".
    """
    query_lower = query.lower()
    
    words = query.split()
    
    # Heuristic 1: Keyword queries
    # Look for quoted phrases or acronyms (all caps, length > 1, optionally with numbers/punctuation)
    has_quotes = '"' in query or "'" in query
    # A simple check for acronyms: any token that is upper case and alphabetical/alphanumeric
    has_acronyms = any(word.isupper() and len(word) > 1 and word.isalpha() for word in words)
    
    if has_quotes or has_acronyms:
        return "keyword"

    # Heuristic 2: Conceptual queries (starts with explicit intent questions)
    if query_lower.startswith(("what is", "explain", "describe", "how does", "tell me about")):
        return "conceptual"

    # Heuristic 3: Vague queries
    if len(words) < 3:
        return "vague"

    # Default to conceptual if broad but not explicitly matching other rules
    return "conceptual"

def choose_strategy(query_type: str) -> Dict[str, Any]:
    """
    Determine the retrieval strategy based on the classified query type.
    """
    if query_type == "keyword":
        return {
            "top_k": 5,
            "bm25_top_k": 50,    # Emphasize lexical matching
            "vector_top_k": 10,  # Less emphasis on semantic
            "rrf_k": 60
        }
    elif query_type == "vague":
        return {
            "top_k": 3,          # Keep it focused
            "bm25_top_k": 10,
            "vector_top_k": 10,
            "rrf_k": 60
        }
    else:  # conceptual or default
        return {
            "top_k": 5,
            "bm25_top_k": 20,
            "vector_top_k": 40,  # Emphasize semantic matching
            "rrf_k": 60
        }
