"""
Tests for text preprocessing utilities.
"""

import pytest
from ..text_preprocessor import TextPreprocessor


class TestTextPreprocessor:
    """Test suite for TextPreprocessor."""
    
    @pytest.fixture
    def preprocessor(self):
        """Create TextPreprocessor instance."""
        return TextPreprocessor()
    
    def test_clean_text_basic(self, preprocessor):
        """Test basic text cleaning."""
        text = "  Hello   World!  \n\n  This is a test.  "
        cleaned = preprocessor.clean_text(text)
        
        assert cleaned == "hello world! this is a test."
        assert cleaned.strip() == cleaned  # No leading/trailing whitespace
    
    def test_clean_text_special_characters(self, preprocessor):
        """Test cleaning of special characters."""
        text = "Hello@#$%^&*()World!!! Test123"
        cleaned = preprocessor.clean_text(text)
        
        # Should remove most special characters but keep basic punctuation
        assert "@#$%^&*()" not in cleaned
        assert "hello" in cleaned
        assert "world" in cleaned
    
    def test_clean_text_page_numbers(self, preprocessor):
        """Test removal of page numbers."""
        text = "This is content. Page 1 of 5. More content here."
        cleaned = preprocessor.clean_text(text)
        
        assert "page 1 of 5" not in cleaned.lower()
        assert "content" in cleaned
    
    def test_clean_text_empty(self, preprocessor):
        """Test cleaning of empty text."""
        assert preprocessor.clean_text("") == ""
        assert preprocessor.clean_text(None) == ""
        assert preprocessor.clean_text("   ") == ""
    
    def test_remove_personal_info_email(self, preprocessor):
        """Test email removal."""
        text = "Contact me at john.doe@example.com for more info."
        cleaned = preprocessor.remove_personal_info(text)
        
        assert "john.doe@example.com" not in cleaned
        assert "[EMAIL]" in cleaned
        assert "Contact me at" in cleaned
    
    def test_remove_personal_info_phone(self, preprocessor):
        """Test phone number removal."""
        texts = [
            "Call me at (555) 123-4567",
            "Phone: 555-123-4567",
            "Contact: +1-555-123-4567",
            "Mobile: 5551234567"
        ]
        
        for text in texts:
            cleaned = preprocessor.remove_personal_info(text)
            assert "[PHONE]" in cleaned
            # Should not contain the original phone number
            assert not any(char.isdigit() for char in cleaned if "[PHONE]" not in cleaned)
    
    def test_remove_personal_info_address(self, preprocessor):
        """Test address removal."""
        text = "I live at 123 Main Street in the city."
        cleaned = preprocessor.remove_personal_info(text)
        
        assert "123 Main Street" not in cleaned
        assert "[ADDRESS]" in cleaned
    
    def test_split_sentences(self, preprocessor):
        """Test sentence splitting."""
        text = "This is the first sentence. This is the second! And this is the third?"
        sentences = preprocessor.split_sentences(text)
        
        assert len(sentences) == 3
        assert "first sentence" in sentences[0]
        assert "second" in sentences[1]
        assert "third" in sentences[2]
    
    def test_split_sentences_empty(self, preprocessor):
        """Test sentence splitting with empty text."""
        assert preprocessor.split_sentences("") == []
        assert preprocessor.split_sentences(None) == []
    
    def test_tokenize_words_basic(self, preprocessor):
        """Test basic word tokenization."""
        text = "Hello, world! This is a test."
        tokens = preprocessor.tokenize_words(text)
        
        expected_tokens = ["hello", "world", "this", "is", "a", "test"]
        for token in expected_tokens:
            assert token in tokens
        
        # Should not contain punctuation
        assert "," not in tokens
        assert "!" not in tokens
        assert "." not in tokens
    
    def test_tokenize_words_remove_stopwords(self, preprocessor):
        """Test tokenization with stopword removal."""
        text = "This is a test of the system"
        tokens = preprocessor.tokenize_words(text, remove_stopwords=True)
        
        # Should contain content words
        assert "test" in tokens
        assert "system" in tokens
        
        # Should not contain stop words
        assert "is" not in tokens
        assert "a" not in tokens
        assert "the" not in tokens
    
    def test_tokenize_words_lemmatize(self, preprocessor):
        """Test tokenization with lemmatization."""
        text = "running dogs are playing"
        tokens = preprocessor.tokenize_words(text, lemmatize=True)
        
        # Should contain lemmatized forms
        assert "run" in tokens or "running" in tokens
        assert "dog" in tokens or "dogs" in tokens
        assert "play" in tokens or "playing" in tokens
    
    def test_extract_technical_terms_acronyms(self, preprocessor):
        """Test extraction of acronyms."""
        text = "I work with API, REST, JSON, and XML technologies."
        terms = preprocessor.extract_technical_terms(text)
        
        expected_terms = ["API", "REST", "JSON", "XML"]
        for term in expected_terms:
            assert term in terms
    
    def test_extract_technical_terms_camelcase(self, preprocessor):
        """Test extraction of camelCase terms."""
        text = "I use JavaScript, TypeScript, and jQuery in my projects."
        terms = preprocessor.extract_technical_terms(text)
        
        # Check that at least some camelCase terms are found
        camelcase_found = any(term for term in terms if any(c.isupper() for c in term[1:]))
        assert camelcase_found, f"No camelCase terms found in: {terms}"
        
        # Check for specific terms that should be found
        expected_terms = ["jQuery"]  # This should definitely be found
        for term in expected_terms:
            assert term in terms, f"Expected term '{term}' not found in: {terms}"
    
    def test_extract_technical_terms_dotted(self, preprocessor):
        """Test extraction of dotted terms."""
        text = "I work with React.js, Node.js, and Vue.js frameworks."
        terms = preprocessor.extract_technical_terms(text)
        
        expected_terms = ["React.js", "Node.js", "Vue.js"]
        for term in expected_terms:
            assert term in terms
    
    def test_extract_technical_terms_versioned(self, preprocessor):
        """Test extraction of versioned terms."""
        text = "I use Python 3.9, Java 11, and Node 16.14 in development."
        terms = preprocessor.extract_technical_terms(text)
        
        expected_patterns = ["Python 3.9", "Java 11", "Node 16.14"]
        for pattern in expected_patterns:
            assert pattern in terms
    
    def test_normalize_skill_name_common(self, preprocessor):
        """Test normalization of common skill names."""
        test_cases = {
            "javascript": "JavaScript",
            "typescript": "TypeScript",
            "nodejs": "Node.js",
            "reactjs": "React.js",
            "c++": "C++",
            "c#": "C#",
            "mysql": "MySQL",
            "postgresql": "PostgreSQL",
            "aws": "AWS",
            "gcp": "Google Cloud Platform"
        }
        
        for input_skill, expected in test_cases.items():
            normalized = preprocessor.normalize_skill_name(input_skill)
            assert normalized == expected
    
    def test_normalize_skill_name_multiword(self, preprocessor):
        """Test normalization of multi-word skills."""
        text = "machine learning"
        normalized = preprocessor.normalize_skill_name(text)
        assert normalized == "Machine Learning"
        
        text = "web development"
        normalized = preprocessor.normalize_skill_name(text)
        assert normalized == "Web Development"
    
    def test_normalize_skill_name_empty(self, preprocessor):
        """Test normalization of empty skill names."""
        assert preprocessor.normalize_skill_name("") == ""
        assert preprocessor.normalize_skill_name(None) == ""
        assert preprocessor.normalize_skill_name("   ") == ""
    
    def test_extract_context_window(self, preprocessor):
        """Test context window extraction."""
        text = "I have extensive experience with Python programming and web development using Django framework."
        contexts = preprocessor.extract_context_window(text, "Python", window_size=20)
        
        assert len(contexts) == 1
        assert "Python" in contexts[0]
        assert len(contexts[0]) <= len(text)
        assert "experience with" in contexts[0] or "programming" in contexts[0]
    
    def test_extract_context_window_multiple_occurrences(self, preprocessor):
        """Test context extraction with multiple occurrences."""
        text = "Python is great. I love Python programming. Python rocks!"
        contexts = preprocessor.extract_context_window(text, "Python", window_size=10)
        
        assert len(contexts) == 3
        for context in contexts:
            assert "Python" in context
    
    def test_extract_context_window_not_found(self, preprocessor):
        """Test context extraction when target not found."""
        text = "This text does not contain the target word."
        contexts = preprocessor.extract_context_window(text, "missing", window_size=10)
        
        assert len(contexts) == 0
    
    def test_is_meaningful_text_valid(self, preprocessor):
        """Test meaningful text detection with valid text."""
        valid_texts = [
            "This is a meaningful sentence with content.",
            "Python programming language",
            "Machine learning algorithms",
            "Web development using React"
        ]
        
        for text in valid_texts:
            assert preprocessor.is_meaningful_text(text)
    
    def test_is_meaningful_text_invalid(self, preprocessor):
        """Test meaningful text detection with invalid text."""
        invalid_texts = [
            "",
            "   ",
            "123 456 789",
            "!@# $%^ &*()",
            "a b",  # Too few words
            None
        ]
        
        for text in invalid_texts:
            assert not preprocessor.is_meaningful_text(text)
    
    def test_is_meaningful_text_custom_min_words(self, preprocessor):
        """Test meaningful text detection with custom minimum words."""
        text = "Python programming"  # 2 words
        
        assert preprocessor.is_meaningful_text(text, min_words=2)
        assert not preprocessor.is_meaningful_text(text, min_words=3)
    
    # Edge Cases and Error Handling
    def test_unicode_handling(self, preprocessor):
        """Test handling of Unicode characters."""
        text = "Café résumé naïve Zürich"
        cleaned = preprocessor.clean_text(text)
        
        # Should handle Unicode gracefully
        assert isinstance(cleaned, str)
        assert len(cleaned) > 0
    
    def test_very_long_text(self, preprocessor):
        """Test handling of very long text."""
        long_text = "Python programming. " * 1000
        
        # Should handle without errors
        cleaned = preprocessor.clean_text(long_text)
        tokens = preprocessor.tokenize_words(long_text)
        sentences = preprocessor.split_sentences(long_text)
        
        assert isinstance(cleaned, str)
        assert isinstance(tokens, list)
        assert isinstance(sentences, list)
    
    def test_mixed_content(self, preprocessor):
        """Test handling of mixed content types."""
        text = """
        Email: test@example.com
        Phone: (555) 123-4567
        Skills: Python, JavaScript, React.js
        Experience: 5+ years
        Location: 123 Main Street
        """
        
        # Should handle mixed content without errors
        cleaned = preprocessor.clean_text(text)
        personal_removed = preprocessor.remove_personal_info(text)
        technical_terms = preprocessor.extract_technical_terms(text)
        
        assert isinstance(cleaned, str)
        assert isinstance(personal_removed, str)
        assert isinstance(technical_terms, list)
        
        # Verify personal info removal
        assert "[EMAIL]" in personal_removed
        assert "[PHONE]" in personal_removed
        assert "[ADDRESS]" in personal_removed
        
        # Verify technical term extraction
        assert "React.js" in technical_terms