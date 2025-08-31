"""
Tests for skill classification module.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch

from ..skill_classifier import SkillClassifier
from ..models import SkillCategory


class TestSkillClassifier:
    """Test suite for SkillClassifier."""
    
    @pytest.fixture
    def classifier(self):
        """Create SkillClassifier instance."""
        classifier = SkillClassifier()
        yield classifier
        classifier.cleanup()
    
    # Test Skill Classification
    @pytest.mark.asyncio
    async def test_classify_skill_direct_lookup(self, classifier):
        """Test direct skill lookup."""
        # Test programming languages
        assert await classifier.classify_skill("python") == SkillCategory.PROGRAMMING_LANGUAGES
        assert await classifier.classify_skill("Python") == SkillCategory.PROGRAMMING_LANGUAGES
        assert await classifier.classify_skill("PYTHON") == SkillCategory.PROGRAMMING_LANGUAGES
        
        # Test frameworks
        assert await classifier.classify_skill("react") == SkillCategory.FRAMEWORKS_LIBRARIES
        assert await classifier.classify_skill("django") == SkillCategory.FRAMEWORKS_LIBRARIES
        
        # Test databases
        assert await classifier.classify_skill("mysql") == SkillCategory.DATABASES
        assert await classifier.classify_skill("postgresql") == SkillCategory.DATABASES
        
        # Test cloud platforms
        assert await classifier.classify_skill("aws") == SkillCategory.CLOUD_PLATFORMS
        assert await classifier.classify_skill("azure") == SkillCategory.CLOUD_PLATFORMS
        
        # Test DevOps tools
        assert await classifier.classify_skill("docker") == SkillCategory.DEVOPS_TOOLS
        assert await classifier.classify_skill("kubernetes") == SkillCategory.DEVOPS_TOOLS
        
        # Test operating systems
        assert await classifier.classify_skill("linux") == SkillCategory.OPERATING_SYSTEMS
        assert await classifier.classify_skill("windows") == SkillCategory.OPERATING_SYSTEMS
        
        # Test soft skills
        assert await classifier.classify_skill("leadership") == SkillCategory.SOFT_SKILLS
        assert await classifier.classify_skill("teamwork") == SkillCategory.SOFT_SKILLS
    
    @pytest.mark.asyncio
    async def test_classify_skill_partial_matching(self, classifier):
        """Test partial skill matching."""
        # Should match partial strings
        assert await classifier.classify_skill("python programming") == SkillCategory.PROGRAMMING_LANGUAGES
        assert await classifier.classify_skill("react development") == SkillCategory.FRAMEWORKS_LIBRARIES
        assert await classifier.classify_skill("mysql database") == SkillCategory.DATABASES
    
    @pytest.mark.asyncio
    async def test_classify_skill_rule_based(self, classifier):
        """Test rule-based classification."""
        # Programming language patterns
        assert await classifier.classify_skill("new programming language") == SkillCategory.PROGRAMMING_LANGUAGES
        assert await classifier.classify_skill("scripting language") == SkillCategory.PROGRAMMING_LANGUAGES
        
        # Framework patterns
        assert await classifier.classify_skill("web framework") == SkillCategory.FRAMEWORKS_LIBRARIES
        assert await classifier.classify_skill("custom library") == SkillCategory.FRAMEWORKS_LIBRARIES
        
        # Database patterns
        assert await classifier.classify_skill("database management") == SkillCategory.DATABASES
        assert await classifier.classify_skill("sql queries") == SkillCategory.DATABASES
        
        # Cloud patterns
        assert await classifier.classify_skill("cloud computing") == SkillCategory.CLOUD_PLATFORMS
        assert await classifier.classify_skill("saas platform") == SkillCategory.CLOUD_PLATFORMS
        
        # DevOps patterns
        assert await classifier.classify_skill("ci/cd pipeline") == SkillCategory.DEVOPS_TOOLS
        assert await classifier.classify_skill("container orchestration") == SkillCategory.DEVOPS_TOOLS
        
        # OS patterns
        assert await classifier.classify_skill("operating system") == SkillCategory.OPERATING_SYSTEMS
        assert await classifier.classify_skill("unix system") == SkillCategory.OPERATING_SYSTEMS
        
        # Soft skills patterns
        assert await classifier.classify_skill("project management") == SkillCategory.SOFT_SKILLS
        assert await classifier.classify_skill("team leadership") == SkillCategory.SOFT_SKILLS
    
    @pytest.mark.asyncio
    async def test_classify_skill_unknown(self, classifier):
        """Test classification of unknown skills."""
        unknown_skills = [
            "completely unknown skill",
            "random text",
            "xyz123",
            ""
        ]
        
        for skill in unknown_skills:
            category = await classifier.classify_skill(skill)
            # Should default to OTHER for unknown skills
            assert category == SkillCategory.OTHER
    
    @pytest.mark.asyncio
    async def test_classify_skill_empty(self, classifier):
        """Test classification of empty skill."""
        assert await classifier.classify_skill("") == SkillCategory.OTHER
        assert await classifier.classify_skill(None) == SkillCategory.OTHER
        assert await classifier.classify_skill("   ") == SkillCategory.OTHER
    
    # Test Batch Classification
    @pytest.mark.asyncio
    async def test_classify_skills_batch(self, classifier):
        """Test batch skill classification."""
        skills = [
            "python",
            "react",
            "mysql",
            "aws",
            "docker",
            "linux",
            "leadership",
            "unknown skill"
        ]
        
        results = await classifier.classify_skills_batch(skills)
        
        assert len(results) == len(skills)
        assert results["python"] == SkillCategory.PROGRAMMING_LANGUAGES
        assert results["react"] == SkillCategory.FRAMEWORKS_LIBRARIES
        assert results["mysql"] == SkillCategory.DATABASES
        assert results["aws"] == SkillCategory.CLOUD_PLATFORMS
        assert results["docker"] == SkillCategory.DEVOPS_TOOLS
        assert results["linux"] == SkillCategory.OPERATING_SYSTEMS
        assert results["leadership"] == SkillCategory.SOFT_SKILLS
        assert results["unknown skill"] == SkillCategory.OTHER
    
    @pytest.mark.asyncio
    async def test_classify_skills_batch_empty(self, classifier):
        """Test batch classification with empty list."""
        results = await classifier.classify_skills_batch([])
        assert results == {}
    
    # Test Category Skills
    def test_get_category_skills(self, classifier):
        """Test getting skills for a category."""
        # Test programming languages
        prog_skills = classifier.get_category_skills(SkillCategory.PROGRAMMING_LANGUAGES)
        assert "python" in prog_skills
        assert "java" in prog_skills
        assert "javascript" in prog_skills
        
        # Test frameworks
        framework_skills = classifier.get_category_skills(SkillCategory.FRAMEWORKS_LIBRARIES)
        assert "react" in framework_skills
        assert "django" in framework_skills
        assert "spring" in framework_skills
        
        # Test unknown category
        unknown_skills = classifier.get_category_skills(SkillCategory.OTHER)
        assert isinstance(unknown_skills, list)
    
    # Test Similar Skills
    def test_get_similar_skills(self, classifier):
        """Test getting similar skills."""
        # Test with known skill
        similar = classifier.get_similar_skills("python", top_k=3)
        assert isinstance(similar, list)
        assert len(similar) <= 3
        assert "python" not in [s.lower() for s in similar]  # Should not include itself
        
        # Should return skills from the same category
        for skill in similar:
            assert skill in classifier.get_category_skills(SkillCategory.PROGRAMMING_LANGUAGES)
    
    def test_get_similar_skills_unknown(self, classifier):
        """Test getting similar skills for unknown skill."""
        similar = classifier.get_similar_skills("unknown_skill_xyz", top_k=3)
        assert isinstance(similar, list)
        # May return empty list or skills from OTHER category
    
    def test_get_similar_skills_custom_k(self, classifier):
        """Test getting similar skills with custom k."""
        similar = classifier.get_similar_skills("python", top_k=5)
        assert len(similar) <= 5
        
        similar = classifier.get_similar_skills("python", top_k=1)
        assert len(similar) <= 1
    
    # Test String Similarity
    def test_calculate_string_similarity(self, classifier):
        """Test string similarity calculation."""
        # Identical strings
        assert classifier._calculate_string_similarity("python", "python") == 1.0
        
        # Similar strings
        sim = classifier._calculate_string_similarity("python", "python3")
        assert 0.5 < sim < 1.0
        
        # Different strings
        sim = classifier._calculate_string_similarity("python", "java")
        assert 0.0 <= sim < 0.5
        
        # Empty strings
        assert classifier._calculate_string_similarity("", "") == 1.0
        assert classifier._calculate_string_similarity("python", "") == 0.0
        assert classifier._calculate_string_similarity("", "python") == 0.0
    
    # Test Similarity-based Classification
    @pytest.mark.asyncio
    async def test_classify_by_similarity(self, classifier):
        """Test similarity-based classification."""
        # Mock the vectorizer to control similarity results
        with patch.object(classifier, 'vectorizer') as mock_vectorizer:
            with patch.object(classifier, 'category_vectors') as mock_vectors:
                # Mock high similarity with programming languages
                mock_vectorizer.transform.return_value = [[0.1, 0.2, 0.3]]
                
                with patch('sklearn.metrics.pairwise.cosine_similarity') as mock_cosine:
                    mock_cosine.return_value = [[0.8, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]
                    
                    category = await classifier._classify_by_similarity("new_programming_lang")
                    assert category == SkillCategory.PROGRAMMING_LANGUAGES
    
    @pytest.mark.asyncio
    async def test_classify_by_similarity_low_score(self, classifier):
        """Test similarity-based classification with low similarity."""
        with patch.object(classifier, 'vectorizer') as mock_vectorizer:
            with patch.object(classifier, 'category_vectors') as mock_vectors:
                mock_vectorizer.transform.return_value = [[0.1, 0.2, 0.3]]
                
                with patch('sklearn.metrics.pairwise.cosine_similarity') as mock_cosine:
                    # All similarities below threshold
                    mock_cosine.return_value = [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]
                    
                    category = await classifier._classify_by_similarity("unknown_skill")
                    assert category == SkillCategory.OTHER
    
    # Test Category Vector Building
    def test_build_category_vectors(self, classifier):
        """Test category vector building."""
        # Vectors should be built during initialization
        assert classifier.vectorizer is not None
        assert classifier.category_vectors is not None
        assert classifier.category_names is not None
        
        # Should have vectors for all categories
        assert len(classifier.category_names) == len(classifier.skill_categories)
    
    # Test Error Handling
    @pytest.mark.asyncio
    async def test_classify_by_similarity_error_handling(self, classifier):
        """Test error handling in similarity classification."""
        with patch.object(classifier, 'vectorizer') as mock_vectorizer:
            # Simulate an error
            mock_vectorizer.transform.side_effect = Exception("Vectorizer error")
            
            category = await classifier._classify_by_similarity("test_skill")
            assert category == SkillCategory.OTHER
    
    # Test Cleanup
    def test_cleanup(self, classifier):
        """Test resource cleanup."""
        # Should not raise any exceptions
        classifier.cleanup()
    
    # Test Case Sensitivity
    @pytest.mark.asyncio
    async def test_case_insensitive_classification(self, classifier):
        """Test case-insensitive skill classification."""
        test_cases = [
            ("Python", "python", "PYTHON"),
            ("JavaScript", "javascript", "JAVASCRIPT"),
            ("React", "react", "REACT"),
            ("MySQL", "mysql", "MYSQL")
        ]
        
        for case1, case2, case3 in test_cases:
            cat1 = await classifier.classify_skill(case1)
            cat2 = await classifier.classify_skill(case2)
            cat3 = await classifier.classify_skill(case3)
            
            assert cat1 == cat2 == cat3
    
    # Test Skill Categories Completeness
    def test_skill_categories_completeness(self, classifier):
        """Test that all skill categories have skills defined."""
        for category in SkillCategory:
            if category != SkillCategory.OTHER:
                skills = classifier.get_category_skills(category)
                assert len(skills) > 0, f"Category {category} has no skills defined"
    
    def test_skill_to_category_mapping(self, classifier):
        """Test skill to category reverse mapping."""
        # Check that reverse mapping is built correctly
        assert len(classifier.skill_to_category) > 0
        
        # Test some known mappings
        assert classifier.skill_to_category.get("python") == SkillCategory.PROGRAMMING_LANGUAGES
        assert classifier.skill_to_category.get("react") == SkillCategory.FRAMEWORKS_LIBRARIES
        assert classifier.skill_to_category.get("mysql") == SkillCategory.DATABASES
    
    # Performance Tests
    @pytest.mark.asyncio
    async def test_classification_performance(self, classifier):
        """Test classification performance with many skills."""
        import time
        
        skills = ["python", "java", "javascript", "react", "angular", "vue"] * 10
        
        start_time = time.time()
        results = await classifier.classify_skills_batch(skills)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Should process quickly
        assert processing_time < 5.0, f"Batch classification took too long: {processing_time}s"
        assert len(results) == len(skills)
    
    # Integration-like Tests
    @pytest.mark.asyncio
    async def test_real_world_skills(self, classifier):
        """Test classification of real-world skill variations."""
        real_world_skills = {
            # Programming languages with variations
            "Python 3.9": SkillCategory.PROGRAMMING_LANGUAGES,
            "JavaScript ES6": SkillCategory.PROGRAMMING_LANGUAGES,
            "C++ programming": SkillCategory.PROGRAMMING_LANGUAGES,
            
            # Frameworks with versions
            "React.js 18": SkillCategory.FRAMEWORKS_LIBRARIES,
            "Django REST Framework": SkillCategory.FRAMEWORKS_LIBRARIES,
            "Spring Boot": SkillCategory.FRAMEWORKS_LIBRARIES,
            
            # Cloud services
            "Amazon Web Services": SkillCategory.CLOUD_PLATFORMS,
            "Google Cloud Platform": SkillCategory.CLOUD_PLATFORMS,
            "Microsoft Azure": SkillCategory.CLOUD_PLATFORMS,
            
            # DevOps tools
            "Docker containers": SkillCategory.DEVOPS_TOOLS,
            "Kubernetes orchestration": SkillCategory.DEVOPS_TOOLS,
            "Jenkins CI/CD": SkillCategory.DEVOPS_TOOLS,
            
            # Databases
            "PostgreSQL database": SkillCategory.DATABASES,
            "MongoDB NoSQL": SkillCategory.DATABASES,
            "Redis caching": SkillCategory.DATABASES,
        }
        
        for skill, expected_category in real_world_skills.items():
            actual_category = await classifier.classify_skill(skill)
            assert actual_category == expected_category, f"Skill '{skill}' classified as {actual_category}, expected {expected_category}"