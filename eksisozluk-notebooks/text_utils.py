"""
Text processing utilities for Ekşi Sözlük analysis.
Provides clean, reusable functions for tokenization and text cleaning.
"""

import re
from typing import List, Set, Optional
from itertools import chain
import pandas as pd
from collections import Counter

def clean_text(text: str) -> str:
    """
    Clean and normalize Turkish text.
    Removes URLs, BKZ references, repeated GÖRSEL patterns, and normalizes whitespace.

    Args:
        text: Raw text string

    Returns:
        Cleaned text with normalized whitespace and punctuation
    """
    if not isinstance(text, str):
        return ""

    # Normalize different apostrophe types to standard apostrophe
    APOSTROPHE_MAP = str.maketrans({
        "'": "'",
        "'": "'",
        "`": "'",
        "´": "'",
        "‛": "'",
    })
    text = text.translate(APOSTROPHE_MAP)

    # Lowercase first
    text = text.lower()

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)

    # Remove BKZ references (bakınız - ekşi sözlük cross-references)
    text = re.sub(r'\bbkz[:.]?\s*', ' ', text, flags=re.IGNORECASE)

    # Remove repeated GÖRSEL patterns (image placeholders)
    text = re.sub(r'(görsel[\s,;/]*){2,}', ' ', text, flags=re.IGNORECASE)

    # Remove mentions and hashtags
    text = re.sub(r'@\w+|#\w+', ' ', text)

    # Keep only Turkish characters (including apostrophes for possessives)
    # This removes punctuation but keeps letters
    text = re.sub(r'[^a-zğüşöçıîâû\'\s]', ' ', text)

    # Normalize multiple spaces to single space
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def get_turkish_stopwords(include_domain_stopwords: bool = True) -> Set[str]:
    """
    Returns a comprehensive set of Turkish stopwords.

    Args:
        include_domain_stopwords: Whether to include domain-specific stopwords

    Returns:
        Set of Turkish stopwords
    """
    # Base Turkish stopwords
    base_stopwords = {
        "acaba", "ama", "ancak", "artık", "aslında", "ayrıca", "bana", "bazı", "belki", "ben",
        "benden", "beni", "benim", "bile", "bir", "birkaç", "birçok", "biri", "birşey",
        "biz", "bizden", "bizi", "bizim", "bu", "buna", "bunda", "bunlar", "bunları", "bunların",
        "bunu", "bunun", "da", "daha", "de", "defa", "diğer", "diğerleri", "diye", "dolayı", "elbette",
        "en", "fakat", "gibi", "hem", "hep", "hepsi", "her", "herkes", "hiç", "için", "ile", "ise", "ister",
        "kadar", "keşke", "ki", "kim", "kimse", "lütfen", "mı", "mi", "mu", "mü", "nasıl", "ne", "neden",
        "nedenle", "nerde", "nerede", "nereden", "nereye", "niye", "o", "olan", "olarak", "oldu",
        "olduğu", "olsa", "olup", "ona", "ondan", "onlar", "onları", "onların", "onu", "onun", "öyle",
        "sanki", "sen", "senden", "seni", "senin", "siz", "sizden", "sizi", "sizin", "şayet", "şimdi",
        "şöyle", "şu", "şuna", "şunda", "şunlar", "şunu", "şunun", "tarafından", "ve", "veya", "ya", "yani",
        "yapılan", "yapmak", "yapılmış", "yapıyor", "yapılmak", "yok", "çok", "çünkü",
        "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz", "on", "var", "nin", "nın",
        "şey", "şeyi", "şeyler", "şeyleri", "şeylerin", "eğer", "değil", "göre",
        # Additional stopwords from your analysis
        "abi", "aga", "hani", "işte", "zaten", "gerçekten", "tam", "dahi", "demek",
        "tabii", "tabi", "yine", "gene", "bence", "sence", "karşı", "doğru", "sonra",
        "önce", "henüz", "belki", "muhtemelen", "kesinlikle", "asla", "hiçbir",
        "herhangi", "birtakım", "pek", "oldukça", "epey", "hayli", "az", "biraz",
        "der", "madem", "mademki", "şayet", "diye", "sadece", "yalnız", "yalnızca",
        "lakin", "hatta", "halbuki",
        # English words and Turkish possessive suffixes that appear as noise
        "in", "ın", "the", "is", "of", "for", "and", "to", "a", "an"
        # edat 
        "üzerine", "üzerinde", "içinde", "altında", "üstünde", "yanında", "önünde", "arkasında",
        "doğusunda", "batısında", "kuzeyinde", "güneyinde", "etrafında", "hakkında",
        "dolayısıyla", "üzerinden", "boyunca", "doğru", "göre", "karşı", "beri", "dek", "kadar", "ötürü", "dolayı", "için",
        "mı", "mi", "mu", "mü", "değil", "ki", "ise", "da", "de", "ve", "ile", "ya", "veya", "hem", "hemde"
        
        # abbreviations
        "vs", "vb", "mr", "mrs", "dr", "prof", "inc", "ltd", "jr", "sr", "co", "tl"
    }

    # Domain-specific stopwords (geopolitical context)
    domain_stopwords = {
        "ülke", "devlet", "dünya", "insan", "millet", "hükümet", "böyle", "aynen"
    }

    # Words to never remove (important for context)
    keep_words = {"kürt", "ermeni"}

    if include_domain_stopwords:
        stopwords = (base_stopwords | domain_stopwords) - keep_words
    else:
        stopwords = base_stopwords - keep_words

    return stopwords

def tokenize(text: str, min_length: int = 2, stopwords: Optional[Set[str]] = None) -> List[str]:
    """
    Tokenize Turkish text with improved suffix handling.
    
    Args:
        text: Input text to tokenize
        min_length: Minimum token length to keep
        stopwords: Optional set of stopwords to filter out
        
    Returns:
        List of tokens with normalized suffixes
    """
    if not isinstance(text, str):
        return []
    
    if stopwords is None:
        stopwords = get_turkish_stopwords()
    
    # First clean the text using existing clean_text function
    text = clean_text(text)
    
    # Handle Turkish suffixes with comprehensive patterns
    suffix_patterns = [
        # Yönelme hali (dative case)
        (r"(\w+)'[yY][EeAa]\b", r"\1"),
        # Belirtme hali (accusative case)
        (r"(\w+)'[yY][IiİıUuÜü]\b", r"\1"),
        # Bulunma hali (locative case)
        (r"(\w+)'[DdTt][EeAa]\b", r"\1"),
        # Ayrılma hali (ablative case)
        (r"(\w+)'[DdTt][EeAa][Nn]\b", r"\1"),
        # İlgi hali (genitive case)
        (r"(\w+)'[Nn][IiİıUuÜü][Nn]\b", r"\1"),
        # İle hali (instrumental case)
        (r"(\w+)'[Ll][EeAa]\b", r"\1"),
        # Çoğul eki (plural suffix)
        (r"(\w+)'[Ll][EeAa][Rr]\b", r"\1"),
        # Kişi ekleri (possessive suffixes)
        (r"(\w+)'[IiİıUuÜü][Mm]\b", r"\1"),
        (r"(\w+)'[Ss][IiİıUuÜü][Nn]\b", r"\1"),
    ]
    
    # Apply all suffix patterns
    for pattern, repl in suffix_patterns:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    
    # Extract tokens with Turkish character support
    tokens = re.findall(r"[a-zğüşöçıîâû]+", text.lower())
    
    # Filter tokens
    filtered_tokens = [
        token for token in tokens
        if len(token) >= min_length
        and token not in stopwords
    ]
    
    return filtered_tokens


def extract_country_terms(topic_titles) -> Set[str]:
    """
    Extract significant terms from topic titles.
    Uses frequency and length-based filtering.
    
    Args:
        topic_titles: Iterable of titles
        
    Returns:
        Set of significant terms
    """
    if hasattr(topic_titles, 'tolist'):
        topic_titles = topic_titles.tolist()
    
    terms = set()
    word_counts = Counter()
    
    # First pass: count all words
    for title in topic_titles:
        words = re.findall(r"[a-zğüşöçıîâû]+", title.lower())
        word_counts.update(words)
    
    # Second pass: add significant terms
    for title in topic_titles:
        words = re.findall(r"[a-zğüşöçıîâû]+", title.lower())
        
        if len(words) == 1:
            # Single word title: add if long enough
            if len(words[0]) > 3:
                terms.add(words[0])
        else:
            # Multi-word title: add frequent enough words
            terms.update(w for w in words 
                        if word_counts[w] >= 2 and len(w) > 3)
    
    return terms


def aggregate_tokens_by_group(df, group_col: str, token_col: str = 'tokens') -> dict:
    """
    Aggregate tokens by a grouping column (e.g., topic, author).

    Args:
        df: DataFrame with tokenized data
        group_col: Column name to group by
        token_col: Column name containing token lists

    Returns:
        Dictionary mapping group values to aggregated token lists
    """
    return (
        df.groupby(group_col)[token_col]
        .apply(lambda series: list(chain.from_iterable(series)))
        .to_dict()
    )
