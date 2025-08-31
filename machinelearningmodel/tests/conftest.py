"""
Pytest configuration and fixtures for ML model tests.
"""

import pytest
import asyncio
import tempfile
import os
from pathlib import Path


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test (may require real models)"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_directory():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def sample_skills_data():
    """Sample skills data for testing."""
    return {
        "programming_languages": [
            "Python", "Java", "JavaScript", "TypeScript", "C++", "C#", "Go", "Rust"
        ],
        "frameworks": [
            "React", "Angular", "Vue", "Django", "Flask", "Spring", "Express"
        ],
        "databases": [
            "MySQL", "PostgreSQL", "MongoDB", "Redis", "Elasticsearch"
        ],
        "cloud_platforms": [
            "AWS", "Azure", "Google Cloud Platform", "Heroku"
        ],
        "devops_tools": [
            "Docker", "Kubernetes", "Jenkins", "GitLab CI", "Terraform"
        ]
    }


@pytest.fixture
def sample_resume_texts():
    """Sample resume texts for testing."""
    return {
        "software_engineer": """
        John Doe
        Senior Software Engineer
        
        EXPERIENCE
        Senior Software Engineer at TechCorp (2020-2023)
        - Developed web applications using Python, Django, and React
        - Worked with PostgreSQL and Redis for data storage
        - Implemented CI/CD pipelines using Docker and Kubernetes
        
        SKILLS
        Programming: Python, JavaScript, TypeScript
        Frameworks: Django, React, Node.js
        Databases: PostgreSQL, MongoDB
        Cloud: AWS, Docker
        """,
        
        "data_scientist": """
        Jane Smith
        Data Scientist
        
        EXPERIENCE
        Data Scientist at DataCorp (2021-2023)
        - Built machine learning models using Python and TensorFlow
        - Analyzed large datasets with Pandas and NumPy
        - Deployed models to AWS using Docker
        
        SKILLS
        Programming: Python, R, SQL
        ML Frameworks: TensorFlow, PyTorch, scikit-learn
        Tools: Jupyter, Git, Docker
        """,
        
        "frontend_developer": """
        Bob Johnson
        Frontend Developer
        
        EXPERIENCE
        Frontend Developer at WebCorp (2019-2023)
        - Built responsive web applications using React and TypeScript
        - Implemented state management with Redux
        - Worked with REST APIs and GraphQL
        
        SKILLS
        Languages: JavaScript, TypeScript, HTML, CSS
        Frameworks: React, Vue.js, Angular
        Tools: Webpack, Babel, ESLint
        """
    }


# Skip integration tests by default unless explicitly requested
def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle integration tests."""
    if config.getoption("--run-integration"):
        return
    
    skip_integration = pytest.mark.skip(reason="need --run-integration option to run")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="run integration tests"
    )
    parser.addoption(
        "--run-slow",
        action="store_true", 
        default=False,
        help="run slow tests"
    )


@pytest.fixture
def mock_models():
    """Mock ML models for testing without requiring actual model downloads."""
    class MockSpacyModel:
        def __call__(self, text):
            # Mock spaCy doc with entities
            class MockDoc:
                def __init__(self, text):
                    self.text = text
                    self.ents = []
                    # Add some mock entities
                    if "Python" in text:
                        self.ents.append(MockEntity("Python", "PRODUCT"))
                    if "React" in text:
                        self.ents.append(MockEntity("React", "PRODUCT"))
                
                @property
                def noun_chunks(self):
                    # Simple mock noun chunks
                    words = self.text.split()
                    return [MockChunk(word) for word in words if len(word) > 3]
            
            return MockDoc(text)
    
    class MockEntity:
        def __init__(self, text, label):
            self.text = text
            self.label_ = label
            self.sent = MockSentence(f"This is a sentence with {text}.")
    
    class MockSentence:
        def __init__(self, text):
            self.text = text
    
    class MockChunk:
        def __init__(self, text):
            self.text = text
    
    class MockSentenceTransformer:
        def encode(self, texts):
            # Return mock embeddings
            import numpy as np
            if isinstance(texts, str):
                texts = [texts]
            return np.random.rand(len(texts), 384)  # Mock 384-dim embeddings
    
    class MockTokenizer:
        def from_pretrained(self, model_name, **kwargs):
            return self
    
    class MockModel:
        def from_pretrained(self, model_name, **kwargs):
            return self
    
    return {
        'spacy_model': MockSpacyModel(),
        'sentence_transformer': MockSentenceTransformer(),
        'tokenizer': MockTokenizer(),
        'model': MockModel()
    }