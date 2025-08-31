#!/usr/bin/env python3
"""
Basic test script for training pipeline components without external dependencies.
"""

import sys
import os
import numpy as np
from datetime import datetime

# Add the machinelearningmodel directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'machinelearningmodel'))

def test_data_preparation():
    """Test basic data preparation functionality."""
    print("Testing data preparation...")
    
    try:
        from training.data_preparation import (
            DataPreprocessor, create_synthetic_training_data
        )
        
        # Test synthetic data creation
        data = create_synthetic_training_data(n_users=10, n_items=5, n_interactions=20)
        assert len(data['users']) == 10
        assert len(data['items']) == 5
        assert len(data['interactions']) == 20
        print("✓ Synthetic data creation works")
        
        # Test data preprocessor
        preprocessor = DataPreprocessor()
        texts = ["Contact me at john@example.com", "I know Python and JavaScript"]
        cleaned = preprocessor.clean_text_data(texts)
        assert "[EMAIL]" in cleaned[0]
        assert "Python" in cleaned[1]
        print("✓ Text cleaning works")
        
        return True
        
    except Exception as e:
        print(f"✗ Data preparation test failed: {e}")
        return False

def test_model_evaluation():
    """Test model evaluation functionality."""
    print("Testing model evaluation...")
    
    try:
        from training.model_evaluator import ClassificationEvaluator, EvaluationMetrics
        
        # Test classification evaluator
        evaluator = ClassificationEvaluator()
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        
        metrics = evaluator.evaluate(y_true, y_pred)
        assert metrics.accuracy is not None
        assert 0 <= metrics.accuracy <= 1
        print("✓ Classification evaluation works")
        
        # Test evaluation metrics creation
        eval_metrics = EvaluationMetrics(accuracy=0.8, precision=0.75)
        assert eval_metrics.accuracy == 0.8
        print("✓ Evaluation metrics creation works")
        
        return True
        
    except Exception as e:
        print(f"✗ Model evaluation test failed: {e}")
        return False

def test_continuous_learning():
    """Test continuous learning functionality."""
    print("Testing continuous learning...")
    
    try:
        from training.continuous_learning import (
            UserFeedback, FeedbackType, OnlineRecommendationModel,
            create_feedback_simulation
        )
        
        # Test feedback creation
        feedback = UserFeedback(
            user_id='user1',
            item_id='item1',
            feedback_type=FeedbackType.EXPLICIT,
            feedback_value=4.0,
            context={'device': 'mobile'},
            timestamp=datetime.now()
        )
        assert feedback.user_id == 'user1'
        print("✓ User feedback creation works")
        
        # Test feedback simulation
        feedback_list = create_feedback_simulation(n_users=5, n_items=3, n_feedback=10)
        assert len(feedback_list) == 10
        print("✓ Feedback simulation works")
        
        # Test online model
        model = OnlineRecommendationModel('test_model')
        assert model.model_id == 'test_model'
        assert not model.is_fitted
        print("✓ Online model creation works")
        
        return True
        
    except Exception as e:
        print(f"✗ Continuous learning test failed: {e}")
        return False

def test_ab_testing():
    """Test A/B testing functionality."""
    print("Testing A/B testing...")
    
    try:
        from training.ab_testing import (
            ExperimentVariant, ExperimentConfig, TrafficSplitMethod,
            create_model_comparison_experiment
        )
        
        # Test variant creation
        variant = ExperimentVariant(
            variant_id='control',
            name='Control',
            description='Control variant',
            model_config={'model': 'baseline'},
            traffic_allocation=0.5,
            is_control=True
        )
        assert variant.variant_id == 'control'
        print("✓ Experiment variant creation works")
        
        # Test model comparison experiment
        model_configs = {
            'model_a': {'name': 'Model A', 'description': 'First model'},
            'model_b': {'name': 'Model B', 'description': 'Second model'}
        }
        experiment_config = create_model_comparison_experiment(model_configs)
        assert len(experiment_config.variants) == 2
        print("✓ Model comparison experiment creation works")
        
        return True
        
    except Exception as e:
        print(f"✗ A/B testing test failed: {e}")
        return False

def test_model_versioning():
    """Test model versioning functionality."""
    print("Testing model versioning...")
    
    try:
        from training.model_versioning import ModelVersion, ModelStatus
        from training.model_trainer import TrainingResult
        from training.model_evaluator import EvaluationMetrics
        
        # Test model version creation
        training_result = TrainingResult(
            model_id='test_model',
            model_type='test',
            training_metrics={'accuracy': 0.8},
            validation_metrics={'accuracy': 0.75},
            test_metrics={'accuracy': 0.77},
            best_epoch=10,
            training_time=100.0,
            hyperparameters={'param1': 1},
            model_path='/tmp/model.pkl',
            metadata={}
        )
        
        eval_metrics = EvaluationMetrics(accuracy=0.77)
        
        model_version = ModelVersion(
            model_id='test_model',
            version='1.0.0',
            model_type='test',
            training_result=training_result,
            evaluation_metrics=eval_metrics,
            model_path='/tmp/model.pkl',
            metadata={},
            created_at=datetime.now(),
            created_by='test',
            status=ModelStatus.TRAINING,
            tags=['test']
        )
        
        assert model_version.model_id == 'test_model'
        assert model_version.status == ModelStatus.TRAINING
        print("✓ Model version creation works")
        
        return True
        
    except Exception as e:
        print(f"✗ Model versioning test failed: {e}")
        return False

def main():
    """Run all basic tests."""
    print("Running basic training pipeline tests...\n")
    
    tests = [
        test_data_preparation,
        test_model_evaluation,
        test_continuous_learning,
        test_ab_testing,
        test_model_versioning
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All basic tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())