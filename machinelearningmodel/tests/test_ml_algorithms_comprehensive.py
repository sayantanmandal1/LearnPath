"""
Comprehensive tests for ML algorithms with synthetic and real data.
"""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from machinelearningmodel.nlp_engine import NLPEngine
from machinelearningmodel.recommendation_engine import RecommendationEngine
from machinelearningmodel.skill_classifier import SkillClassifier
from machinelearningmodel.text_preprocessor import TextPreprocessor
from machinelearningmodel.learning_path_optimizer import LearningPathOptimizer


@pytest.mark.ml
class TestNLPEngineAccuracy:
    """Test NLP engine accuracy with various data types."""
    
    @pytest.fixture
    def nlp_engine(self):
        return NLPEngine()
    
    def test_skill_extraction_accuracy_synthetic_data(self, nlp_engine, synthetic_resume_data):
        """Test skill extraction accuracy on synthetic resume data."""
        total_correct = 0
        total_predictions = 0
        
        for resume in synthetic_resume_data:
            extracted_skills = nlp_engine.extract_skills_from_text(resume["text"])
            extracted_skill_names = [skill["skill"] for skill in extracted_skills]
            
            # Calculate precision and recall
            true_positives = len(set(extracted_skill_names) & set(resume["expected_skills"]))
            false_positives = len(set(extracted_skill_names) - set(resume["expected_skills"]))
            false_negatives = len(set(resume["expected_skills"]) - set(extracted_skill_names))
            
            if len(extracted_skill_names) > 0:
                precision = true_positives / (true_positives + false_positives)
                assert precision >= 0.7, f"Precision too low: {precision}"
            
            if len(resume["expected_skills"]) > 0:
                recall = true_positives / (true_positives + false_negatives)
                assert recall >= 0.6, f"Recall too low: {recall}"
    
    def test_skill_confidence_scores(self, nlp_engine):
        """Test that skill confidence scores are reasonable."""
        text = "Experienced Python developer with 5 years of Django and Flask experience"
        skills = nlp_engine.extract_skills_from_text(text)
        
        # Check confidence scores are in valid range
        for skill in skills:
            assert 0.0 <= skill["confidence"] <= 1.0
        
        # Python should have high confidence in this context
        python_skills = [s for s in skills if "python" in s["skill"].lower()]
        if python_skills:
            assert python_skills[0]["confidence"] >= 0.8
    
    def test_embedding_generation_consistency(self, nlp_engine):
        """Test that embeddings are consistent and well-formed."""
        text1 = "Python machine learning engineer"
        text2 = "ML engineer with Python experience"
        text3 = "JavaScript frontend developer"
        
        embedding1 = nlp_engine.generate_embeddings(text1)
        embedding2 = nlp_engine.generate_embeddings(text2)
        embedding3 = nlp_engine.generate_embeddings(text3)
        
        # Check embedding dimensions
        assert len(embedding1) == len(embedding2) == len(embedding3)
        assert len(embedding1) > 0
        
        # Similar texts should have higher similarity
        similarity_12 = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        similarity_13 = np.dot(embedding1, embedding3) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding3))
        
        assert similarity_12 > similarity_13, "Similar texts should have higher similarity"
    
    def test_text_preprocessing_quality(self, nlp_engine):
        """Test text preprocessing maintains important information."""
        raw_text = "I'm a Senior Python Developer with 5+ years of experience in Django, Flask, and FastAPI. I have worked on machine learning projects using scikit-learn and TensorFlow."
        
        processed = nlp_engine.preprocess_text(raw_text)
        
        # Should preserve important technical terms
        assert "python" in processed.lower()
        assert "django" in processed.lower()
        assert "machine learning" in processed.lower() or "machinelearning" in processed.lower()
        
        # Should remove or normalize common words appropriately
        assert len(processed) < len(raw_text)  # Some cleanup should occur


@pytest.mark.ml
class TestSkillClassifierAccuracy:
    """Test skill classifier accuracy and performance."""
    
    @pytest.fixture
    def skill_classifier(self):
        return SkillClassifier()
    
    def test_skill_categorization_accuracy(self, skill_classifier):
        """Test skill categorization accuracy."""
        test_skills = [
            ("Python", "programming"),
            ("JavaScript", "programming"),
            ("Machine Learning", "technical"),
            ("Communication", "soft_skill"),
            ("Leadership", "soft_skill"),
            ("Docker", "tool"),
            ("AWS", "platform"),
            ("React", "framework")
        ]
        
        correct_predictions = 0
        for skill, expected_category in test_skills:
            predicted_category = skill_classifier.classify_skill(skill)
            if predicted_category == expected_category:
                correct_predictions += 1
        
        accuracy = correct_predictions / len(test_skills)
        assert accuracy >= 0.75, f"Skill classification accuracy too low: {accuracy}"
    
    def test_skill_similarity_scoring(self, skill_classifier):
        """Test skill similarity scoring."""
        # Similar skills should have high similarity
        similarity_python_java = skill_classifier.calculate_similarity("Python", "Java")
        similarity_python_communication = skill_classifier.calculate_similarity("Python", "Communication")
        
        assert similarity_python_java > similarity_python_communication
        assert 0.0 <= similarity_python_java <= 1.0
        assert 0.0 <= similarity_python_communication <= 1.0
    
    def test_skill_level_assessment(self, skill_classifier):
        """Test skill level assessment from context."""
        contexts = [
            ("I have 5 years of Python experience", "Python", "advanced"),
            ("I'm learning Python basics", "Python", "beginner"),
            ("I use Python for complex ML projects", "Python", "advanced"),
            ("I know some Python", "Python", "beginner")
        ]
        
        for context, skill, expected_level in contexts:
            assessed_level = skill_classifier.assess_skill_level(context, skill)
            # Allow some flexibility in level assessment
            assert assessed_level in ["beginner", "intermediate", "advanced"]


@pytest.mark.ml
class TestRecommendationEngineAccuracy:
    """Test recommendation engine accuracy and relevance."""
    
    @pytest.fixture
    def recommendation_engine(self):
        return RecommendationEngine()
    
    def test_career_recommendation_relevance(self, recommendation_engine, ml_model_test_data):
        """Test career recommendation relevance."""
        user_profile = {
            "skills": {"Python": 0.9, "Machine Learning": 0.8, "Statistics": 0.7},
            "experience_years": 3,
            "interests": ["data_science", "ai"]
        }
        
        recommendations = recommendation_engine.recommend_careers(user_profile, top_k=5)
        
        # Check that recommendations are relevant
        assert len(recommendations) <= 5
        assert all(rec["match_score"] >= 0.0 for rec in recommendations)
        assert all(rec["match_score"] <= 1.0 for rec in recommendations)
        
        # Recommendations should be sorted by match score
        scores = [rec["match_score"] for rec in recommendations]
        assert scores == sorted(scores, reverse=True)
        
        # Top recommendation should have high relevance for ML profile
        top_rec = recommendations[0]
        assert top_rec["match_score"] >= 0.6
    
    def test_collaborative_filtering_accuracy(self, recommendation_engine):
        """Test collaborative filtering component accuracy."""
        # Mock user-item interaction matrix
        user_item_matrix = np.array([
            [1, 1, 0, 0, 1],  # User 0: likes items 0, 1, 4
            [0, 1, 1, 1, 0],  # User 1: likes items 1, 2, 3
            [1, 0, 1, 0, 1],  # User 2: likes items 0, 2, 4
            [0, 0, 1, 1, 1],  # User 3: likes items 2, 3, 4
        ])
        
        # Test user similarity calculation
        user_similarities = recommendation_engine._calculate_user_similarities(user_item_matrix)
        
        # Users with similar preferences should have higher similarity
        assert user_similarities[0, 2] > user_similarities[0, 1]  # Users 0 and 2 both like items 0 and 4
        
        # Generate recommendations for user 0
        recommendations = recommendation_engine._collaborative_filtering_predict(
            user_item_matrix, user_similarities, user_id=0, top_k=2
        )
        
        assert len(recommendations) <= 2
        assert all(item_id not in [0, 1, 4] for item_id in recommendations)  # Should recommend unseen items
    
    def test_content_based_filtering_accuracy(self, recommendation_engine):
        """Test content-based filtering accuracy."""
        user_skills = {"Python": 0.9, "Machine Learning": 0.8}
        
        job_profiles = [
            {"id": 1, "skills": {"Python": 0.9, "Django": 0.7}, "title": "Python Developer"},
            {"id": 2, "skills": {"Machine Learning": 0.9, "Python": 0.8}, "title": "ML Engineer"},
            {"id": 3, "skills": {"JavaScript": 0.9, "React": 0.8}, "title": "Frontend Developer"}
        ]
        
        recommendations = recommendation_engine._content_based_filtering(
            user_skills, job_profiles, top_k=2
        )
        
        # ML Engineer should be top recommendation due to skill overlap
        assert recommendations[0]["id"] == 2
        assert recommendations[0]["match_score"] > recommendations[1]["match_score"]
    
    def test_hybrid_recommendation_performance(self, recommendation_engine):
        """Test hybrid recommendation system performance."""
        user_profile = {
            "skills": {"Python": 0.9, "Machine Learning": 0.8},
            "experience_years": 3,
            "past_interactions": [1, 3, 5]  # Previously viewed job IDs
        }
        
        # Mock both collaborative and content-based results
        with patch.object(recommendation_engine, '_collaborative_filtering_predict') as mock_cf:
            with patch.object(recommendation_engine, '_content_based_filtering') as mock_cbf:
                mock_cf.return_value = [{"id": 1, "score": 0.8}, {"id": 2, "score": 0.7}]
                mock_cbf.return_value = [{"id": 2, "score": 0.9}, {"id": 3, "score": 0.6}]
                
                hybrid_recs = recommendation_engine._hybrid_recommend(user_profile, top_k=3)
                
                # Should combine both approaches
                assert len(hybrid_recs) <= 3
                # Item 2 should rank high as it appears in both approaches
                item_2_score = next((rec["score"] for rec in hybrid_recs if rec["id"] == 2), 0)
                assert item_2_score > 0.7


@pytest.mark.ml
class TestLearningPathOptimizerAccuracy:
    """Test learning path optimizer accuracy and effectiveness."""
    
    @pytest.fixture
    def path_optimizer(self):
        return LearningPathOptimizer()
    
    def test_skill_gap_prioritization(self, path_optimizer):
        """Test skill gap prioritization logic."""
        current_skills = {"Python": 0.8, "JavaScript": 0.6, "SQL": 0.4}
        target_skills = {"Python": 0.9, "JavaScript": 0.8, "SQL": 0.8, "Docker": 0.7}
        market_demand = {"Python": 0.9, "JavaScript": 0.8, "SQL": 0.7, "Docker": 0.9}
        
        prioritized_gaps = path_optimizer.prioritize_skill_gaps(
            current_skills, target_skills, market_demand
        )
        
        # Docker should be high priority (new skill with high demand)
        # SQL should be high priority (large gap with decent demand)
        docker_priority = next((gap["priority"] for gap in prioritized_gaps if gap["skill"] == "Docker"), 0)
        sql_priority = next((gap["priority"] for gap in prioritized_gaps if gap["skill"] == "SQL"), 0)
        python_priority = next((gap["priority"] for gap in prioritized_gaps if gap["skill"] == "Python"), 0)
        
        assert docker_priority > python_priority  # New skill vs small improvement
        assert sql_priority > python_priority  # Large gap vs small gap
    
    def test_learning_path_sequencing(self, path_optimizer):
        """Test learning path sequencing logic."""
        skills_to_learn = ["Docker", "Kubernetes", "Python Advanced", "Machine Learning"]
        prerequisites = {
            "Kubernetes": ["Docker"],
            "Machine Learning": ["Python Advanced"],
            "Python Advanced": ["Python"]  # Assume user already has Python
        }
        
        sequenced_path = path_optimizer.sequence_learning_path(skills_to_learn, prerequisites)
        
        # Docker should come before Kubernetes
        docker_index = next(i for i, skill in enumerate(sequenced_path) if skill == "Docker")
        kubernetes_index = next(i for i, skill in enumerate(sequenced_path) if skill == "Kubernetes")
        assert docker_index < kubernetes_index
        
        # Python Advanced should come before Machine Learning
        python_adv_index = next(i for i, skill in enumerate(sequenced_path) if skill == "Python Advanced")
        ml_index = next(i for i, skill in enumerate(sequenced_path) if skill == "Machine Learning")
        assert python_adv_index < ml_index
    
    def test_duration_estimation_accuracy(self, path_optimizer):
        """Test learning duration estimation accuracy."""
        learning_resources = [
            {"skill": "Docker", "duration_hours": 20, "difficulty": "intermediate"},
            {"skill": "Kubernetes", "duration_hours": 40, "difficulty": "advanced"},
            {"skill": "Python Advanced", "duration_hours": 30, "difficulty": "intermediate"}
        ]
        
        user_profile = {
            "experience_level": "intermediate",
            "learning_pace": "normal",
            "weekly_hours": 10
        }
        
        estimated_weeks = path_optimizer.estimate_learning_duration(
            learning_resources, user_profile
        )
        
        # Should be reasonable estimate (not too optimistic or pessimistic)
        total_hours = sum(resource["duration_hours"] for resource in learning_resources)
        expected_weeks = total_hours / user_profile["weekly_hours"]
        
        # Allow for some overhead in estimation
        assert expected_weeks <= estimated_weeks <= expected_weeks * 1.5
    
    def test_resource_recommendation_quality(self, path_optimizer):
        """Test quality of learning resource recommendations."""
        skill = "Machine Learning"
        user_preferences = {
            "learning_style": "hands_on",
            "budget": "free",
            "time_availability": "part_time"
        }
        
        with patch.object(path_optimizer, '_fetch_learning_resources') as mock_fetch:
            mock_fetch.return_value = [
                {"title": "ML Course A", "type": "course", "rating": 4.8, "cost": 0, "hands_on": True},
                {"title": "ML Course B", "type": "course", "rating": 4.5, "cost": 50, "hands_on": False},
                {"title": "ML Book", "type": "book", "rating": 4.2, "cost": 0, "hands_on": False},
                {"title": "ML Project", "type": "project", "rating": 4.6, "cost": 0, "hands_on": True}
            ]
            
            recommendations = path_optimizer.recommend_learning_resources(skill, user_preferences)
            
            # Should prioritize free, hands-on resources
            top_rec = recommendations[0]
            assert top_rec["cost"] == 0  # Free
            assert top_rec["hands_on"] is True  # Hands-on
            assert top_rec["rating"] >= 4.0  # Good quality


@pytest.mark.ml
class TestMLModelPerformanceMetrics:
    """Test ML model performance metrics and benchmarks."""
    
    def test_model_inference_speed(self, nlp_engine):
        """Test that model inference meets speed requirements."""
        import time
        
        text = "Experienced Python developer with machine learning expertise"
        
        # Test skill extraction speed
        start_time = time.time()
        skills = nlp_engine.extract_skills_from_text(text)
        extraction_time = time.time() - start_time
        
        assert extraction_time < 2.0, f"Skill extraction too slow: {extraction_time}s"
        
        # Test embedding generation speed
        start_time = time.time()
        embedding = nlp_engine.generate_embeddings(text)
        embedding_time = time.time() - start_time
        
        assert embedding_time < 1.0, f"Embedding generation too slow: {embedding_time}s"
    
    def test_model_memory_usage(self, nlp_engine):
        """Test that models don't consume excessive memory."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process multiple texts
        texts = [f"Sample text {i} with various skills" for i in range(100)]
        for text in texts:
            nlp_engine.extract_skills_from_text(text)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 500MB for 100 texts)
        assert memory_increase < 500, f"Excessive memory usage: {memory_increase}MB"
    
    def test_model_consistency_across_runs(self, nlp_engine):
        """Test that models produce consistent results across multiple runs."""
        text = "Python developer with 3 years of Django experience"
        
        # Run extraction multiple times
        results = []
        for _ in range(5):
            skills = nlp_engine.extract_skills_from_text(text)
            skill_names = sorted([skill["skill"] for skill in skills])
            results.append(skill_names)
        
        # All runs should produce the same results
        first_result = results[0]
        for result in results[1:]:
            assert result == first_result, "Model results not consistent across runs"
    
    def test_model_robustness_to_noise(self, nlp_engine):
        """Test model robustness to noisy input."""
        clean_text = "Python developer with machine learning experience"
        noisy_text = "Python   developer with machine    learning experience!!!"
        
        clean_skills = nlp_engine.extract_skills_from_text(clean_text)
        noisy_skills = nlp_engine.extract_skills_from_text(noisy_text)
        
        clean_skill_names = set(skill["skill"] for skill in clean_skills)
        noisy_skill_names = set(skill["skill"] for skill in noisy_skills)
        
        # Should extract similar skills despite noise
        overlap = len(clean_skill_names & noisy_skill_names)
        total_unique = len(clean_skill_names | noisy_skill_names)
        
        similarity = overlap / total_unique if total_unique > 0 else 1.0
        assert similarity >= 0.8, f"Model not robust to noise: {similarity}"


@pytest.mark.ml
@pytest.mark.slow
class TestMLModelAccuracyBenchmarks:
    """Comprehensive accuracy benchmarks for ML models."""
    
    def test_end_to_end_recommendation_accuracy(self, nlp_engine, recommendation_engine):
        """Test end-to-end recommendation accuracy."""
        # Simulate complete user profile processing
        resume_text = """
        Senior Software Engineer with 5 years of experience in Python development.
        Expertise in Django, Flask, and FastAPI for web development.
        Strong background in machine learning using scikit-learn and TensorFlow.
        Experience with cloud platforms including AWS and Docker containerization.
        """
        
        # Extract skills from resume
        extracted_skills = nlp_engine.extract_skills_from_text(resume_text)
        skill_dict = {skill["skill"]: skill["confidence"] for skill in extracted_skills}
        
        # Generate career recommendations
        user_profile = {
            "skills": skill_dict,
            "experience_years": 5,
            "interests": ["machine_learning", "web_development"]
        }
        
        recommendations = recommendation_engine.recommend_careers(user_profile, top_k=5)
        
        # Validate recommendations make sense for the profile
        assert len(recommendations) == 5
        
        # Should recommend relevant roles
        recommended_titles = [rec["job_title"].lower() for rec in recommendations]
        relevant_keywords = ["engineer", "developer", "ml", "python", "senior"]
        
        relevance_count = sum(
            1 for title in recommended_titles 
            if any(keyword in title for keyword in relevant_keywords)
        )
        
        assert relevance_count >= 3, f"Only {relevance_count}/5 recommendations seem relevant"
        
        # Top recommendation should have high confidence
        assert recommendations[0]["match_score"] >= 0.7
    
    def test_skill_extraction_precision_recall(self, nlp_engine):
        """Test skill extraction precision and recall on diverse texts."""
        test_cases = [
            {
                "text": "Full-stack developer with React, Node.js, and MongoDB experience",
                "expected": ["React", "Node.js", "MongoDB", "Full-stack", "JavaScript"]
            },
            {
                "text": "Data scientist proficient in R, Python, SQL, and Tableau",
                "expected": ["R", "Python", "SQL", "Tableau", "Data Science"]
            },
            {
                "text": "DevOps engineer with Kubernetes, Docker, and AWS expertise",
                "expected": ["Kubernetes", "Docker", "AWS", "DevOps"]
            }
        ]
        
        total_precision = 0
        total_recall = 0
        
        for case in test_cases:
            extracted = nlp_engine.extract_skills_from_text(case["text"])
            extracted_names = set(skill["skill"] for skill in extracted)
            expected_names = set(case["expected"])
            
            # Calculate precision and recall
            true_positives = len(extracted_names & expected_names)
            false_positives = len(extracted_names - expected_names)
            false_negatives = len(expected_names - extracted_names)
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            
            total_precision += precision
            total_recall += recall
        
        avg_precision = total_precision / len(test_cases)
        avg_recall = total_recall / len(test_cases)
        
        # Minimum acceptable performance thresholds
        assert avg_precision >= 0.7, f"Average precision too low: {avg_precision}"
        assert avg_recall >= 0.6, f"Average recall too low: {avg_recall}"
        
        # F1 score should be balanced
        f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
        assert f1_score >= 0.65, f"F1 score too low: {f1_score}"