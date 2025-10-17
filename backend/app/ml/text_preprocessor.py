"""
Text preprocessing utilities for NLP processing.
"""

import re
import string
from typing import List, Optional
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
def download_nltk_data():
    """Download NLTK data with error handling."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        try:
            nltk.download('punkt', quiet=True)
        except Exception:
            pass  # Fail silently if download fails

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        try:
            nltk.download('stopwords', quiet=True)
        except Exception:
            pass  # Fail silently if download fails

    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        try:
            nltk.download('wordnet', quiet=True)
        except Exception:
            pass  # Fail silently if download fails

# Try to download NLTK data
download_nltk_data()


class TextPreprocessor:
    """
    Text preprocessing utilities for cleaning and normalizing text.
    """
    
    def __init__(self):
        """Initialize the text preprocessor."""
        self.lemmatizer = WordNetLemmatizer()
        
        # Try to load stopwords, fallback to basic set if not available
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            # Fallback to basic English stopwords
            self.stop_words = {
                'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
                'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
                'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
                'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
                'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
                'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
                'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after',
                'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
                'further', 'then', 'once'
            }
        
        # Common resume noise patterns
        self.noise_patterns = [
            r'\b(?:page\s+\d+|\d+\s+of\s+\d+)\b',  # Page numbers
            r'\b(?:confidential|proprietary)\b',    # Confidential markers
            r'[^\w\s\-\.\,\;\:\!\?\(\)]',          # Special characters except basic punctuation
            r'\s+',                                 # Multiple whitespace
        ]
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text for processing.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Convert to lowercase for processing
        cleaned = text.lower()
        
        # Remove noise patterns
        for pattern in self.noise_patterns:
            if pattern == r'\s+':
                cleaned = re.sub(pattern, ' ', cleaned)
            else:
                cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
        # Remove extra whitespace and normalize
        cleaned = ' '.join(cleaned.split())
        
        return cleaned.strip()
    
    def remove_personal_info(self, text: str) -> str:
        """
        Remove personal information from text.
        
        Args:
            text: Text to clean
            
        Returns:
            Text with personal info removed
        """
        # Email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
        
        # Phone numbers
        text = re.sub(r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b', '[PHONE]', text)
        
        # Addresses (basic pattern)
        text = re.sub(r'\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd)\b', '[ADDRESS]', text, flags=re.IGNORECASE)
        
        return text
    
    def split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        if not text:
            return []
        
        try:
            sentences = sent_tokenize(text)
            return [s.strip() for s in sentences if s.strip()]
        except LookupError:
            # Fallback to simple sentence splitting
            sentences = re.split(r'[.!?]+', text)
            return [s.strip() for s in sentences if s.strip()]
    
    def tokenize_words(self, text: str, remove_stopwords: bool = False, lemmatize: bool = False) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Text to tokenize
            remove_stopwords: Whether to remove stop words
            lemmatize: Whether to lemmatize words
            
        Returns:
            List of tokens
        """
        if not text:
            return []
        
        try:
            # Tokenize
            tokens = word_tokenize(text.lower())
        except LookupError:
            # Fallback to simple tokenization
            tokens = re.findall(r'\b\w+\b', text.lower())
        
        # Remove punctuation
        tokens = [token for token in tokens if token not in string.punctuation]
        
        # Remove stop words if requested
        if remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]
        
        # Lemmatize if requested
        if lemmatize:
            try:
                tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
            except LookupError:
                # Skip lemmatization if WordNet is not available
                pass
        
        return tokens
    
    def extract_technical_terms(self, text: str) -> List[str]:
        """
        Extract technical terms and acronyms from text.
        
        Args:
            text: Text to process
            
        Returns:
            List of technical terms
        """
        technical_terms = []
        
        # Extract acronyms (2-10 uppercase letters)
        acronyms = re.findall(r'\b[A-Z]{2,10}\b', text)
        technical_terms.extend(acronyms)
        
        # Extract camelCase terms
        camel_case = re.findall(r'\b[a-z]+[A-Z][a-zA-Z]*\b', text)
        technical_terms.extend(camel_case)
        
        # Extract terms with dots (e.g., React.js, Node.js)
        dotted_terms = re.findall(r'\b[A-Za-z]+\.[A-Za-z]+\b', text)
        technical_terms.extend(dotted_terms)
        
        # Extract version numbers with terms
        versioned_terms = re.findall(r'\b[A-Za-z]+\s+\d+(?:\.\d+)*\b', text)
        technical_terms.extend(versioned_terms)
        
        return list(set(technical_terms))  # Remove duplicates
    
    def normalize_skill_name(self, skill: str) -> str:
        """
        Normalize skill names for consistency.
        
        Args:
            skill: Raw skill name
            
        Returns:
            Normalized skill name
        """
        if not skill:
            return ""
        
        # Basic cleaning
        normalized = skill.strip()
        
        # Common normalizations
        normalizations = {
            'javascript': 'JavaScript',
            'typescript': 'TypeScript',
            'nodejs': 'Node.js',
            'reactjs': 'React.js',
            'vuejs': 'Vue.js',
            'angularjs': 'Angular.js',
            'c++': 'C++',
            'c#': 'C#',
            'mysql': 'MySQL',
            'postgresql': 'PostgreSQL',
            'mongodb': 'MongoDB',
            'aws': 'AWS',
            'gcp': 'Google Cloud Platform',
            'azure': 'Microsoft Azure',
        }
        
        normalized_lower = normalized.lower()
        if normalized_lower in normalizations:
            return normalizations[normalized_lower]
        
        # Capitalize first letter of each word for multi-word skills
        if ' ' in normalized:
            return ' '.join(word.capitalize() for word in normalized.split())
        
        return normalized
    
    def extract_context_window(self, text: str, target: str, window_size: int = 100) -> List[str]:
        """
        Extract context windows around target terms.
        
        Args:
            text: Source text
            target: Target term to find
            window_size: Size of context window in characters
            
        Returns:
            List of context windows
        """
        contexts = []
        
        # Find all occurrences of target
        for match in re.finditer(re.escape(target), text, re.IGNORECASE):
            start = max(0, match.start() - window_size)
            end = min(len(text), match.end() + window_size)
            context = text[start:end].strip()
            contexts.append(context)
        
        return contexts
    
    def is_meaningful_text(self, text: str, min_words: int = 3) -> bool:
        """
        Check if text contains meaningful content.
        
        Args:
            text: Text to check
            min_words: Minimum number of words required
            
        Returns:
            True if text is meaningful
        """
        if not text or not text.strip():
            return False
        
        # Check word count
        words = self.tokenize_words(text, remove_stopwords=True)
        if len(words) < min_words:
            return False
        
        # Check if it's not just numbers or special characters
        if re.match(r'^[\d\s\W]+$', text):
            return False
        
        return True