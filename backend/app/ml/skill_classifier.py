"""
Skill classification module using BERT and rule-based approaches.
"""

import logging
from typing import Dict, List, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

try:
    from .models import SkillCategory
except ImportError:
    from models import SkillCategory

logger = logging.getLogger(__name__)


class SkillClassifier:
    """
    Classifier for categorizing skills into different categories.
    """
    
    def __init__(self, model_cache_dir: str = "./models"):
        """
        Initialize the skill classifier.
        
        Args:
            model_cache_dir: Directory to cache models
        """
        self.model_cache_dir = model_cache_dir
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Skill category definitions
        self.skill_categories = {
            SkillCategory.PROGRAMMING_LANGUAGES: [
                'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'go', 'rust',
                'ruby', 'php', 'swift', 'kotlin', 'scala', 'r', 'matlab', 'perl',
                'shell', 'bash', 'powershell', 'sql', 'html', 'css'
            ],
            SkillCategory.FRAMEWORKS_LIBRARIES: [
                'react', 'angular', 'vue', 'django', 'flask', 'spring', 'express',
                'laravel', 'rails', 'asp.net', 'tensorflow', 'pytorch', 'scikit-learn',
                'pandas', 'numpy', 'opencv', 'keras', 'jquery', 'bootstrap', 'tailwind'
            ],
            SkillCategory.DATABASES: [
                'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'cassandra',
                'oracle', 'sqlite', 'dynamodb', 'neo4j', 'influxdb', 'couchdb'
            ],
            SkillCategory.CLOUD_PLATFORMS: [
                'aws', 'azure', 'gcp', 'google cloud', 'heroku', 'digitalocean',
                'linode', 'vultr', 'cloudflare', 'vercel', 'netlify'
            ],
            SkillCategory.DEVOPS_TOOLS: [
                'docker', 'kubernetes', 'jenkins', 'gitlab ci', 'github actions',
                'terraform', 'ansible', 'chef', 'puppet', 'vagrant', 'helm',
                'prometheus', 'grafana', 'elk stack', 'nagios'
            ],
            SkillCategory.OPERATING_SYSTEMS: [
                'linux', 'windows', 'macos', 'ubuntu', 'centos', 'debian',
                'red hat', 'fedora', 'arch linux', 'freebsd'
            ],
            SkillCategory.SOFT_SKILLS: [
                'leadership', 'communication', 'teamwork', 'problem solving',
                'project management', 'agile', 'scrum', 'mentoring', 'presentation',
                'negotiation', 'time management', 'critical thinking'
            ],
            SkillCategory.TECHNICAL: [
                'git', 'svn', 'rest api', 'graphql', 'microservices', 'websockets',
                'oauth', 'jwt', 'ssl', 'tls', 'tcp/ip', 'http', 'dns', 'load balancing'
            ]
        }
        
        # Build reverse lookup
        self.skill_to_category = {}
        for category, skills in self.skill_categories.items():
            for skill in skills:
                self.skill_to_category[skill.lower()] = category
        
        # TF-IDF vectorizer for similarity matching
        self.vectorizer = None
        self.category_vectors = None
        self._build_category_vectors()
        
        logger.info("Skill classifier initialized")
    
    def _build_category_vectors(self):
        """Build TF-IDF vectors for each skill category."""
        category_texts = []
        category_names = []
        
        for category, skills in self.skill_categories.items():
            category_text = ' '.join(skills)
            category_texts.append(category_text)
            category_names.append(category)
        
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        self.category_vectors = self.vectorizer.fit_transform(category_texts)
        self.category_names = category_names
    
    async def classify_skill(self, skill_name: str) -> SkillCategory:
        """
        Classify a skill into a category.
        
        Args:
            skill_name: Name of the skill to classify
            
        Returns:
            SkillCategory enum value
        """
        if not skill_name:
            return SkillCategory.OTHER
        
        skill_lower = skill_name.lower().strip()
        
        # Direct lookup first
        if skill_lower in self.skill_to_category:
            return self.skill_to_category[skill_lower]
        
        # Partial matching
        for known_skill, category in self.skill_to_category.items():
            if known_skill in skill_lower or skill_lower in known_skill:
                return category
        
        # Similarity-based classification
        category = await self._classify_by_similarity(skill_name)
        if category != SkillCategory.OTHER:
            return category
        
        # Rule-based classification
        return self._classify_by_rules(skill_name)
    
    async def _classify_by_similarity(self, skill_name: str) -> SkillCategory:
        """Classify skill using TF-IDF similarity."""
        def classify():
            try:
                skill_vector = self.vectorizer.transform([skill_name.lower()])
                similarities = cosine_similarity(skill_vector, self.category_vectors)[0]
                
                max_similarity = np.max(similarities)
                if max_similarity > 0.3:  # Threshold for similarity
                    best_category_idx = np.argmax(similarities)
                    return self.category_names[best_category_idx]
                
                return SkillCategory.OTHER
            except Exception as e:
                logger.warning(f"Error in similarity classification: {e}")
                return SkillCategory.OTHER
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, classify)
    
    def _classify_by_rules(self, skill_name: str) -> SkillCategory:
        """Classify skill using rule-based approach."""
        skill_lower = skill_name.lower()
        
        # Programming language patterns
        if any(pattern in skill_lower for pattern in [
            'programming', 'language', 'script', 'code'
        ]):
            return SkillCategory.PROGRAMMING_LANGUAGES
        
        # Framework/library patterns
        if any(pattern in skill_lower for pattern in [
            'framework', 'library', 'lib', '.js', '.py', 'api'
        ]):
            return SkillCategory.FRAMEWORKS_LIBRARIES
        
        # Database patterns
        if any(pattern in skill_lower for pattern in [
            'database', 'db', 'sql', 'nosql', 'query'
        ]):
            return SkillCategory.DATABASES
        
        # Cloud patterns
        if any(pattern in skill_lower for pattern in [
            'cloud', 'aws', 'azure', 'gcp', 'saas', 'paas', 'iaas'
        ]):
            return SkillCategory.CLOUD_PLATFORMS
        
        # DevOps patterns
        if any(pattern in skill_lower for pattern in [
            'devops', 'ci/cd', 'deployment', 'container', 'orchestration'
        ]):
            return SkillCategory.DEVOPS_TOOLS
        
        # OS patterns
        if any(pattern in skill_lower for pattern in [
            'operating system', 'os', 'linux', 'windows', 'unix'
        ]):
            return SkillCategory.OPERATING_SYSTEMS
        
        # Soft skills patterns
        if any(pattern in skill_lower for pattern in [
            'management', 'leadership', 'communication', 'team', 'agile', 'scrum'
        ]):
            return SkillCategory.SOFT_SKILLS
        
        return SkillCategory.OTHER
    
    async def classify_skills_batch(self, skills: List[str]) -> Dict[str, SkillCategory]:
        """
        Classify multiple skills at once.
        
        Args:
            skills: List of skill names
            
        Returns:
            Dictionary mapping skill names to categories
        """
        results = {}
        
        # Process skills in parallel
        tasks = [self.classify_skill(skill) for skill in skills]
        categories = await asyncio.gather(*tasks)
        
        for skill, category in zip(skills, categories):
            results[skill] = category
        
        return results
    
    def get_category_skills(self, category: SkillCategory) -> List[str]:
        """
        Get all known skills for a category.
        
        Args:
            category: Skill category
            
        Returns:
            List of skills in the category
        """
        return self.skill_categories.get(category, [])
    
    def get_similar_skills(self, skill_name: str, top_k: int = 5) -> List[str]:
        """
        Get similar skills to the given skill.
        
        Args:
            skill_name: Input skill name
            top_k: Number of similar skills to return
            
        Returns:
            List of similar skill names
        """
        skill_lower = skill_name.lower()
        similar_skills = []
        
        # Find skills in the same category using synchronous classification
        # Direct lookup first
        if skill_lower in self.skill_to_category:
            category = self.skill_to_category[skill_lower]
        else:
            # Use rule-based classification as fallback
            category = self._classify_by_rules(skill_name)
        
        category_skills = self.get_category_skills(category)
        
        # Calculate similarity scores
        skill_similarities = []
        for candidate_skill in category_skills:
            if candidate_skill.lower() != skill_lower:
                # Simple similarity based on common substrings
                similarity = self._calculate_string_similarity(skill_lower, candidate_skill.lower())
                skill_similarities.append((candidate_skill, similarity))
        
        # Sort by similarity and return top_k
        skill_similarities.sort(key=lambda x: x[1], reverse=True)
        similar_skills = [skill for skill, _ in skill_similarities[:top_k]]
        
        return similar_skills
    
    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        """Calculate simple string similarity."""
        # Jaccard similarity on character bigrams
        def get_bigrams(s):
            return set(s[i:i+2] for i in range(len(s)-1))
        
        bigrams1 = get_bigrams(str1)
        bigrams2 = get_bigrams(str2)
        
        if not bigrams1 and not bigrams2:
            return 1.0
        if not bigrams1 or not bigrams2:
            return 0.0
        
        intersection = len(bigrams1.intersection(bigrams2))
        union = len(bigrams1.union(bigrams2))
        
        return intersection / union if union > 0 else 0.0
    
    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
        logger.info("Skill classifier cleaned up")