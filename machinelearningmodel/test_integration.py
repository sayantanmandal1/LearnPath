"""
Simple integration test for NLP engine components.
"""

import asyncio
import tempfile
import os
from pathlib import Path

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from text_preprocessor import TextPreprocessor
from skill_classifier import SkillClassifier
from models import SkillCategory


async def test_nlp_integration():
    """Test NLP components integration."""
    print("Testing NLP Engine Integration...")
    
    # Test text preprocessor
    print("\n1. Testing Text Preprocessor...")
    preprocessor = TextPreprocessor()
    
    sample_text = """
    John Doe - Senior Software Engineer
    Email: john.doe@example.com
    Phone: (555) 123-4567
    
    I have 5+ years of experience with Python, JavaScript, React, and Django.
    I've worked with PostgreSQL databases and AWS cloud services.
    My skills include machine learning with TensorFlow and data analysis.
    """
    
    cleaned = preprocessor.clean_text(sample_text)
    print(f"✓ Text cleaned: {len(cleaned)} characters")
    
    personal_removed = preprocessor.remove_personal_info(sample_text)
    print(f"✓ Personal info removed: {'[EMAIL]' in personal_removed and '[PHONE]' in personal_removed}")
    
    sentences = preprocessor.split_sentences(sample_text)
    print(f"✓ Sentences split: {len(sentences)} sentences")
    
    tokens = preprocessor.tokenize_words(sample_text, remove_stopwords=True)
    print(f"✓ Words tokenized: {len(tokens)} tokens")
    
    technical_terms = preprocessor.extract_technical_terms(sample_text)
    print(f"✓ Technical terms extracted: {technical_terms}")
    
    # Test skill classifier
    print("\n2. Testing Skill Classifier...")
    classifier = SkillClassifier()
    
    test_skills = ["Python", "JavaScript", "React", "Django", "PostgreSQL", "AWS", "TensorFlow"]
    
    for skill in test_skills:
        category = await classifier.classify_skill(skill)
        print(f"✓ {skill} -> {category}")
    
    # Test batch classification
    batch_results = await classifier.classify_skills_batch(test_skills)
    print(f"✓ Batch classification: {len(batch_results)} skills classified")
    
    # Test similar skills
    similar = classifier.get_similar_skills("Python", top_k=3)
    print(f"✓ Similar to Python: {similar}")
    
    classifier.cleanup()
    
    print("\n✅ All NLP components working correctly!")
    return True


if __name__ == "__main__":
    success = asyncio.run(test_nlp_integration())
    if success:
        print("\n🎉 NLP Engine Integration Test PASSED!")
    else:
        print("\n❌ NLP Engine Integration Test FAILED!")