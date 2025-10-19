"""
Text processing utilities for Ekşi Sözlük analysis.
Provides clean, reusable functions for tokenization and text cleaning.
"""

import re
from typing import List, Set, Optional
from itertools import chain


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


def tokenize(text: str, stopwords: Optional[Set[str]] = None, min_length: int = 2) -> List[str]:
    """
    Tokenize text with stopword removal and length filtering.
    Uses regex pattern to extract Turkish words (including possessives with apostrophes).

    Args:
        text: Input text string
        stopwords: Set of stopwords to remove (defaults to Turkish stopwords)
        min_length: Minimum token length to keep

    Returns:
        List of filtered tokens
    """
    if stopwords is None:
        stopwords = get_turkish_stopwords()

    # Pattern to match Turkish words (with optional apostrophe for possessives)
    TOKEN_PATTERN = re.compile(r"[a-zğüşöçıîâû]+(?:'[a-zğüşöçıîâû]+)?", flags=re.IGNORECASE)

    cleaned = clean_text(text)
    tokens = TOKEN_PATTERN.findall(cleaned)

    # Filter stopwords and short tokens
    tokens = [
        tok for tok in tokens
        if tok not in stopwords and len(tok) >= min_length
    ]

    return tokens


def extract_country_terms(topic_titles) -> Set[str]:
    """
    Extract complete country names from topic titles for filtering.
    Only extracts FULL country names, not fragments from multi-word names.

    For example:
    - "el salvador" → adds only "salvador" (not "el" which is too common)
    - "güney kore" → adds only "kore" (not "güney" which means "south")
    - "fransa" → adds "fransa"

    Args:
        topic_titles: Iterable of topic titles (country names) - can be list, array, or Series

    Returns:
        Set of country terms to filter (single-word countries and significant parts of multi-word names)
    """
    # Convert to list if it's a numpy array or pandas Series
    if hasattr(topic_titles, 'tolist'):
        topic_titles = topic_titles.tolist()

    country_terms = set()

    # Words that are directional/descriptive and shouldn't be filtered alone
    # These are common words that appear in multi-word country names
    directional_words = {'el', 'güney', 'kuzey', 'yeni', 'suudi', 'gürcü', 'bosna'}

    for title in topic_titles:
        normalized_title = title.lower().strip()
        # Extract individual words from the title
        words = re.findall(r"[a-zğüşöçıîâû]+", normalized_title)

        if len(words) == 1:
            # Single-word country: add it
            country_terms.add(words[0])
        else:
            # Multi-word country: only add the significant (non-directional) parts
            for word in words:
                if word not in directional_words:
                    country_terms.add(word)

    return country_terms


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
