"""
Training data preparation and feature engineering pipelines.

This module handles data preprocessing, feature engineering, and dataset creation
for training recommendation models and NLP components.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import json
import pickle
from pathlib import Path
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp

from ..models import SkillExtraction, ResumeData, SkillCategory


logger = logging.getLogger(__name__)


@dataclass
class TrainingDataset:
    """Container for training dataset."""
    X_train: Union[np.ndarray, sp.csr_matrix]
    X_val: Union[np.ndarray, sp.csr_matrix]
    X_test: Union[np.ndarray, sp.csr_matrix]
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]
    metadata: Dict[str, Any]


@dataclass
class UserItemMatrix:
    """User-item interaction matrix for collaborative filtering."""
    matrix: sp.csr_matrix
    user_ids: List[str]
    item_ids: List[str]
    user_id_to_idx: Dict[str, int]
    item_id_to_idx: Dict[str, int]
    metadata: Dict[str, Any]


class DataPreprocessor:
    """Data preprocessing and cleaning utilities."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.scalers = {}
        self.encoders = {}
        self.vectorizers = {}
        self.fitted = False
        
    def clean_text_data(self, texts: List[str]) -> List[str]:
        """Clean and normalize text data."""
        cleaned_texts = []
        
        for text in texts:
            if not isinstance(text, str):
                text = str(text)
            
            # Remove extra whitespace
            text = ' '.join(text.split())
            
            # Remove personal information patterns
            import re
            # Email patterns
            text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
            # Phone patterns
            text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)
            # Address patterns (simplified)
            text = re.sub(r'\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln)\b', '[ADDRESS]', text, flags=re.IGNORECASE)
            
            # Normalize common variations
            text = re.sub(r'\bjavascript\b', 'JavaScript', text, flags=re.IGNORECASE)
            text = re.sub(r'\bpython\b', 'Python', text, flags=re.IGNORECASE)
            text = re.sub(r'\breact\.?js\b', 'React', text, flags=re.IGNORECASE)
            text = re.sub(r'\bnode\.?js\b', 'Node.js', text, flags=re.IGNORECASE)
            
            cleaned_texts.append(text)
        
        return cleaned_texts
    
    def extract_numerical_features(self, data: List[Dict[str, Any]]) -> np.ndarray:
        """Extract numerical features from structured data."""
        features = []
        feature_names = []
        
        for record in data:
            record_features = []
            
            # Experience years
            exp_years = record.get('experience_years', 0)
            record_features.append(exp_years)
            if 'experience_years' not in feature_names:
                feature_names.append('experience_years')
            
            # Number of skills
            skills_count = len(record.get('skills', []))
            record_features.append(skills_count)
            if 'skills_count' not in feature_names:
                feature_names.append('skills_count')
            
            # Education level (encoded)
            education_level = record.get('education_level', 'bachelor')
            education_mapping = {'high_school': 1, 'associate': 2, 'bachelor': 3, 'master': 4, 'phd': 5}
            record_features.append(education_mapping.get(education_level, 3))
            if 'education_level_encoded' not in feature_names:
                feature_names.append('education_level_encoded')
            
            # GitHub stats
            github_stats = record.get('github_stats', {})
            record_features.extend([
                github_stats.get('public_repos', 0),
                github_stats.get('followers', 0),
                github_stats.get('following', 0),
                github_stats.get('total_commits', 0)
            ])
            if 'github_repos' not in feature_names:
                feature_names.extend(['github_repos', 'github_followers', 'github_following', 'github_commits'])
            
            # LeetCode stats
            leetcode_stats = record.get('leetcode_stats', {})
            record_features.extend([
                leetcode_stats.get('problems_solved', 0),
                leetcode_stats.get('contest_rating', 0),
                leetcode_stats.get('acceptance_rate', 0.0)
            ])
            if 'leetcode_problems' not in feature_names:
                feature_names.extend(['leetcode_problems', 'leetcode_rating', 'leetcode_acceptance'])
            
            features.append(record_features)
        
        self.numerical_feature_names = feature_names
        return np.array(features)
    
    def create_skill_features(self, user_skills: List[Dict[str, Any]], 
                            skill_taxonomy: Dict[str, Any]) -> Tuple[np.ndarray, List[str]]:
        """Create skill-based features using one-hot encoding and skill embeddings."""
        all_skills = set()
        for user in user_skills:
            all_skills.update(user.get('skills', []))
        
        skill_list = sorted(list(all_skills))
        skill_features = []
        
        for user in user_skills:
            user_skill_vector = []
            user_skills_set = set(user.get('skills', []))
            
            # One-hot encoding for skills
            for skill in skill_list:
                user_skill_vector.append(1 if skill in user_skills_set else 0)
            
            # Skill category features
            skill_categories = {}
            for skill in user_skills_set:
                category = skill_taxonomy.get(skill, {}).get('category', 'other')
                skill_categories[category] = skill_categories.get(category, 0) + 1
            
            # Add category counts
            for category in SkillCategory:
                user_skill_vector.append(skill_categories.get(category.value, 0))
            
            skill_features.append(user_skill_vector)
        
        feature_names = skill_list + [f"category_{cat.value}" for cat in SkillCategory]
        return np.array(skill_features), feature_names
    
    def fit_transform(self, data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Fit preprocessors and transform data."""
        logger.info("Fitting data preprocessors")
        
        processed_data = {}
        
        # Process numerical features
        if 'numerical_data' in data:
            numerical_features = self.extract_numerical_features(data['numerical_data'])
            scaler = StandardScaler()
            processed_data['numerical'] = scaler.fit_transform(numerical_features)
            self.scalers['numerical'] = scaler
        
        # Process text features
        if 'text_data' in data:
            cleaned_texts = self.clean_text_data(data['text_data'])
            vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
            processed_data['text'] = vectorizer.fit_transform(cleaned_texts)
            self.vectorizers['text'] = vectorizer
        
        # Process skill features
        if 'skill_data' in data and 'skill_taxonomy' in data:
            skill_features, skill_feature_names = self.create_skill_features(
                data['skill_data'], data['skill_taxonomy']
            )
            processed_data['skills'] = skill_features
            self.skill_feature_names = skill_feature_names
        
        # Process categorical features
        if 'categorical_data' in data:
            categorical_features = []
            for column, values in data['categorical_data'].items():
                encoder = LabelEncoder()
                encoded_values = encoder.fit_transform(values)
                categorical_features.append(encoded_values.reshape(-1, 1))
                self.encoders[column] = encoder
            
            if categorical_features:
                processed_data['categorical'] = np.hstack(categorical_features)
        
        self.fitted = True
        return processed_data
    
    def transform(self, data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Transform new data using fitted preprocessors."""
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before transforming data")
        
        processed_data = {}
        
        # Transform numerical features
        if 'numerical_data' in data and 'numerical' in self.scalers:
            numerical_features = self.extract_numerical_features(data['numerical_data'])
            processed_data['numerical'] = self.scalers['numerical'].transform(numerical_features)
        
        # Transform text features
        if 'text_data' in data and 'text' in self.vectorizers:
            cleaned_texts = self.clean_text_data(data['text_data'])
            processed_data['text'] = self.vectorizers['text'].transform(cleaned_texts)
        
        # Transform skill features
        if 'skill_data' in data and hasattr(self, 'skill_feature_names'):
            skill_features, _ = self.create_skill_features(
                data['skill_data'], data.get('skill_taxonomy', {})
            )
            processed_data['skills'] = skill_features
        
        # Transform categorical features
        if 'categorical_data' in data and self.encoders:
            categorical_features = []
            for column, values in data['categorical_data'].items():
                if column in self.encoders:
                    encoded_values = self.encoders[column].transform(values)
                    categorical_features.append(encoded_values.reshape(-1, 1))
            
            if categorical_features:
                processed_data['categorical'] = np.hstack(categorical_features)
        
        return processed_data


class FeatureEngineer:
    """Advanced feature engineering for recommendation systems."""
    
    def __init__(self):
        self.interaction_features = {}
        self.temporal_features = {}
        self.embedding_features = {}
    
    def create_interaction_features(self, user_features: np.ndarray, 
                                  item_features: np.ndarray) -> np.ndarray:
        """Create user-item interaction features."""
        logger.info("Creating interaction features")
        
        # Element-wise multiplication
        interaction_mult = user_features * item_features
        
        # Cosine similarity
        user_norm = np.linalg.norm(user_features, axis=1, keepdims=True)
        item_norm = np.linalg.norm(item_features, axis=1, keepdims=True)
        cosine_sim = np.sum(user_features * item_features, axis=1, keepdims=True) / (user_norm * item_norm + 1e-8)
        
        # Euclidean distance
        euclidean_dist = np.linalg.norm(user_features - item_features, axis=1, keepdims=True)
        
        # Manhattan distance
        manhattan_dist = np.sum(np.abs(user_features - item_features), axis=1, keepdims=True)
        
        return np.hstack([interaction_mult, cosine_sim, euclidean_dist, manhattan_dist])
    
    def create_temporal_features(self, timestamps: List[datetime], 
                               reference_date: Optional[datetime] = None) -> np.ndarray:
        """Create temporal features from timestamps."""
        if reference_date is None:
            reference_date = datetime.now()
        
        temporal_features = []
        
        for timestamp in timestamps:
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)
            
            # Days since reference
            days_since = (reference_date - timestamp).days
            
            # Day of week
            day_of_week = timestamp.weekday()
            
            # Month
            month = timestamp.month
            
            # Hour (if time is available)
            hour = timestamp.hour
            
            # Season
            season = (month - 1) // 3  # 0: Winter, 1: Spring, 2: Summer, 3: Fall
            
            temporal_features.append([days_since, day_of_week, month, hour, season])
        
        return np.array(temporal_features)
    
    def create_statistical_features(self, user_history: Dict[str, List[float]]) -> np.ndarray:
        """Create statistical features from user interaction history."""
        statistical_features = []
        
        for user_id, ratings in user_history.items():
            if not ratings:
                stats = [0, 0, 0, 0, 0, 0]  # Default values
            else:
                ratings_array = np.array(ratings)
                stats = [
                    np.mean(ratings_array),      # Mean rating
                    np.std(ratings_array),       # Standard deviation
                    np.median(ratings_array),    # Median rating
                    np.min(ratings_array),       # Min rating
                    np.max(ratings_array),       # Max rating
                    len(ratings_array)           # Number of ratings
                ]
            
            statistical_features.append(stats)
        
        return np.array(statistical_features)


class UserItemMatrixBuilder:
    """Build user-item interaction matrices for collaborative filtering."""
    
    def __init__(self, min_interactions: int = 5):
        self.min_interactions = min_interactions
    
    def build_matrix(self, interactions: List[Dict[str, Any]], 
                    implicit_feedback: bool = False) -> UserItemMatrix:
        """
        Build user-item interaction matrix from interaction data.
        
        Args:
            interactions: List of interaction records with user_id, item_id, rating
            implicit_feedback: Whether to treat as implicit feedback (binary)
        """
        logger.info(f"Building user-item matrix from {len(interactions)} interactions")
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(interactions)
        
        # Filter users and items with minimum interactions
        user_counts = df['user_id'].value_counts()
        item_counts = df['item_id'].value_counts()
        
        valid_users = user_counts[user_counts >= self.min_interactions].index
        valid_items = item_counts[item_counts >= self.min_interactions].index
        
        df_filtered = df[df['user_id'].isin(valid_users) & df['item_id'].isin(valid_items)]
        
        logger.info(f"Filtered to {len(valid_users)} users and {len(valid_items)} items")
        
        # Create mappings
        user_ids = sorted(df_filtered['user_id'].unique())
        item_ids = sorted(df_filtered['item_id'].unique())
        
        user_id_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
        item_id_to_idx = {item_id: idx for idx, item_id in enumerate(item_ids)}
        
        # Build matrix
        n_users = len(user_ids)
        n_items = len(item_ids)
        
        row_indices = []
        col_indices = []
        data = []
        
        for _, row in df_filtered.iterrows():
            user_idx = user_id_to_idx[row['user_id']]
            item_idx = item_id_to_idx[row['item_id']]
            rating = 1.0 if implicit_feedback else row.get('rating', 1.0)
            
            row_indices.append(user_idx)
            col_indices.append(item_idx)
            data.append(rating)
        
        matrix = sp.csr_matrix((data, (row_indices, col_indices)), shape=(n_users, n_items))
        
        metadata = {
            'n_users': n_users,
            'n_items': n_items,
            'n_interactions': len(data),
            'sparsity': 1.0 - (len(data) / (n_users * n_items)),
            'implicit_feedback': implicit_feedback,
            'min_interactions': self.min_interactions
        }
        
        return UserItemMatrix(
            matrix=matrix,
            user_ids=user_ids,
            item_ids=item_ids,
            user_id_to_idx=user_id_to_idx,
            item_id_to_idx=item_id_to_idx,
            metadata=metadata
        )
    
    def add_negative_samples(self, matrix: UserItemMatrix, 
                           negative_ratio: float = 1.0) -> UserItemMatrix:
        """Add negative samples for implicit feedback training."""
        if not matrix.metadata.get('implicit_feedback', False):
            logger.warning("Adding negative samples to explicit feedback matrix")
        
        logger.info(f"Adding negative samples with ratio {negative_ratio}")
        
        # Get positive interactions
        positive_interactions = set()
        coo_matrix = matrix.matrix.tocoo()
        for user_idx, item_idx in zip(coo_matrix.row, coo_matrix.col):
            positive_interactions.add((user_idx, item_idx))
        
        # Sample negative interactions
        n_positive = len(positive_interactions)
        n_negative = int(n_positive * negative_ratio)
        
        negative_interactions = set()
        while len(negative_interactions) < n_negative:
            user_idx = np.random.randint(0, matrix.matrix.shape[0])
            item_idx = np.random.randint(0, matrix.matrix.shape[1])
            
            if (user_idx, item_idx) not in positive_interactions:
                negative_interactions.add((user_idx, item_idx))
        
        # Create new matrix with negative samples
        row_indices = list(coo_matrix.row) + [user_idx for user_idx, _ in negative_interactions]
        col_indices = list(coo_matrix.col) + [item_idx for _, item_idx in negative_interactions]
        data = list(coo_matrix.data) + [0.0] * len(negative_interactions)
        
        new_matrix = sp.csr_matrix(
            (data, (row_indices, col_indices)), 
            shape=matrix.matrix.shape
        )
        
        # Update metadata
        new_metadata = matrix.metadata.copy()
        new_metadata['n_interactions'] = len(data)
        new_metadata['negative_samples'] = len(negative_interactions)
        
        return UserItemMatrix(
            matrix=new_matrix,
            user_ids=matrix.user_ids,
            item_ids=matrix.item_ids,
            user_id_to_idx=matrix.user_id_to_idx,
            item_id_to_idx=matrix.item_id_to_idx,
            metadata=new_metadata
        )


class TrainingDatasetBuilder:
    """Build training datasets for different ML tasks."""
    
    def __init__(self, test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42):
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
    
    def build_recommendation_dataset(self, user_item_matrix: UserItemMatrix,
                                   user_features: Optional[np.ndarray] = None,
                                   item_features: Optional[np.ndarray] = None) -> TrainingDataset:
        """Build dataset for recommendation model training."""
        logger.info("Building recommendation training dataset")
        
        # Convert sparse matrix to coordinate format for easier manipulation
        coo_matrix = user_item_matrix.matrix.tocoo()
        
        # Create feature matrix and target vector
        n_interactions = len(coo_matrix.data)
        features = []
        targets = coo_matrix.data
        
        for i in range(n_interactions):
            user_idx = coo_matrix.row[i]
            item_idx = coo_matrix.col[i]
            
            # Basic features: user_id, item_id (one-hot encoded)
            user_onehot = np.zeros(user_item_matrix.matrix.shape[0])
            user_onehot[user_idx] = 1
            
            item_onehot = np.zeros(user_item_matrix.matrix.shape[1])
            item_onehot[item_idx] = 1
            
            feature_vector = np.concatenate([user_onehot, item_onehot])
            
            # Add user features if available
            if user_features is not None:
                feature_vector = np.concatenate([feature_vector, user_features[user_idx]])
            
            # Add item features if available
            if item_features is not None:
                feature_vector = np.concatenate([feature_vector, item_features[item_idx]])
            
            features.append(feature_vector)
        
        X = np.array(features)
        y = np.array(targets)
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        val_size_adjusted = self.val_size / (1 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=self.random_state
        )
        
        # Feature names
        feature_names = []
        feature_names.extend([f"user_{i}" for i in range(user_item_matrix.matrix.shape[0])])
        feature_names.extend([f"item_{i}" for i in range(user_item_matrix.matrix.shape[1])])
        
        if user_features is not None:
            feature_names.extend([f"user_feat_{i}" for i in range(user_features.shape[1])])
        
        if item_features is not None:
            feature_names.extend([f"item_feat_{i}" for i in range(item_features.shape[1])])
        
        metadata = {
            'n_users': user_item_matrix.matrix.shape[0],
            'n_items': user_item_matrix.matrix.shape[1],
            'n_interactions': n_interactions,
            'train_size': len(X_train),
            'val_size': len(X_val),
            'test_size': len(X_test),
            'feature_dim': X.shape[1]
        }
        
        return TrainingDataset(
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            feature_names=feature_names,
            metadata=metadata
        )
    
    def build_skill_classification_dataset(self, skill_data: List[Dict[str, Any]]) -> TrainingDataset:
        """Build dataset for skill classification training."""
        logger.info("Building skill classification training dataset")
        
        texts = []
        labels = []
        
        for record in skill_data:
            text = record.get('text', '')
            skill_labels = record.get('skills', [])
            
            texts.append(text)
            labels.append(skill_labels)
        
        # For multi-label classification, we need to create binary targets
        all_skills = set()
        for skill_list in labels:
            all_skills.update(skill_list)
        
        skill_to_idx = {skill: idx for idx, skill in enumerate(sorted(all_skills))}
        
        # Create binary label matrix
        y = np.zeros((len(labels), len(all_skills)))
        for i, skill_list in enumerate(labels):
            for skill in skill_list:
                y[i, skill_to_idx[skill]] = 1
        
        # Use TF-IDF for text features
        vectorizer = TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1, 3))
        X = vectorizer.fit_transform(texts)
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        val_size_adjusted = self.val_size / (1 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=self.random_state
        )
        
        feature_names = vectorizer.get_feature_names_out().tolist()
        
        metadata = {
            'n_samples': len(texts),
            'n_features': X.shape[1],
            'n_skills': len(all_skills),
            'skill_to_idx': skill_to_idx,
            'vectorizer': vectorizer,
            'train_size': X_train.shape[0],
            'val_size': X_val.shape[0],
            'test_size': X_test.shape[0]
        }
        
        return TrainingDataset(
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            feature_names=feature_names,
            metadata=metadata
        )
    
    def save_dataset(self, dataset: TrainingDataset, filepath: str):
        """Save training dataset to disk."""
        logger.info(f"Saving training dataset to {filepath}")
        
        with open(filepath, 'wb') as f:
            pickle.dump(dataset, f)
    
    def load_dataset(self, filepath: str) -> TrainingDataset:
        """Load training dataset from disk."""
        logger.info(f"Loading training dataset from {filepath}")
        
        with open(filepath, 'rb') as f:
            return pickle.load(f)


def create_synthetic_training_data(n_users: int = 1000, n_items: int = 500, 
                                 n_interactions: int = 10000) -> Dict[str, Any]:
    """Create synthetic training data for testing purposes."""
    logger.info(f"Creating synthetic training data: {n_users} users, {n_items} items, {n_interactions} interactions")
    
    np.random.seed(42)
    
    # Generate user data
    users = []
    for i in range(n_users):
        user = {
            'user_id': f'user_{i}',
            'experience_years': np.random.randint(0, 20),
            'education_level': np.random.choice(['bachelor', 'master', 'phd']),
            'skills': np.random.choice(['Python', 'Java', 'JavaScript', 'React', 'SQL', 'AWS'], 
                                     size=np.random.randint(1, 6), replace=False).tolist(),
            'github_stats': {
                'public_repos': np.random.randint(0, 100),
                'followers': np.random.randint(0, 1000),
                'following': np.random.randint(0, 500),
                'total_commits': np.random.randint(0, 5000)
            },
            'leetcode_stats': {
                'problems_solved': np.random.randint(0, 1000),
                'contest_rating': np.random.randint(800, 2500),
                'acceptance_rate': np.random.uniform(0.3, 0.9)
            }
        }
        users.append(user)
    
    # Generate item (job) data
    items = []
    job_titles = ['Software Engineer', 'Data Scientist', 'Product Manager', 'DevOps Engineer', 'Frontend Developer']
    for i in range(n_items):
        item = {
            'item_id': f'job_{i}',
            'title': np.random.choice(job_titles),
            'required_skills': np.random.choice(['Python', 'Java', 'JavaScript', 'React', 'SQL', 'AWS'], 
                                              size=np.random.randint(2, 5), replace=False).tolist(),
            'experience_required': np.random.randint(0, 10),
            'salary_range': (np.random.randint(50000, 80000), np.random.randint(80000, 150000)),
            'description': f'Job description for {np.random.choice(job_titles)}'
        }
        items.append(item)
    
    # Generate interactions
    interactions = []
    for i in range(n_interactions):
        user_id = f'user_{np.random.randint(0, n_users)}'
        item_id = f'job_{np.random.randint(0, n_items)}'
        rating = np.random.uniform(1, 5)
        
        interaction = {
            'user_id': user_id,
            'item_id': item_id,
            'rating': rating,
            'timestamp': datetime.now() - timedelta(days=np.random.randint(0, 365))
        }
        interactions.append(interaction)
    
    # Generate skill taxonomy
    skill_taxonomy = {
        'Python': {'category': 'programming_languages', 'difficulty': 'intermediate'},
        'Java': {'category': 'programming_languages', 'difficulty': 'intermediate'},
        'JavaScript': {'category': 'programming_languages', 'difficulty': 'beginner'},
        'React': {'category': 'frameworks_libraries', 'difficulty': 'intermediate'},
        'SQL': {'category': 'databases', 'difficulty': 'beginner'},
        'AWS': {'category': 'cloud_platforms', 'difficulty': 'advanced'}
    }
    
    return {
        'users': users,
        'items': items,
        'interactions': interactions,
        'skill_taxonomy': skill_taxonomy
    }