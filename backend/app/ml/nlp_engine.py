"""
NLP Processing Engine for Resume Parsing and Skill Extraction

This module provides comprehensive NLP capabilities including:
- PDF/DOC resume parsing
- Skill extraction using spaCy and custom NER models
- BERT-based skill classification
- Semantic embedding generation
- Text preprocessing and cleaning utilities
"""

import logging
import re
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor

import fitz  # PyMuPDF
from docx import Document
import spacy
from spacy.tokens import Doc
import torch
from transformers import AutoTokenizer, AutoModel, pipeline
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from .text_preprocessor import TextPreprocessor
    from .skill_classifier import SkillClassifier
    from .models import ResumeData, SkillExtraction, SkillCategory
except ImportError:
    from text_preprocessor import TextPreprocessor
    from skill_classifier import SkillClassifier
    from models import ResumeData, SkillExtraction, SkillCategory

logger = logging.getLogger(__name__)


class NLPEngine:
    """
    Main NLP processing engine for resume parsing and skill extraction.
    """
    
    def __init__(self, model_cache_dir: str = "./models"):
        """
        Initialize the NLP engine with required models.
        
        Args:
            model_cache_dir: Directory to cache downloaded models
        """
        self.model_cache_dir = Path(model_cache_dir)
        self.model_cache_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.text_preprocessor = TextPreprocessor()
        self.skill_classifier = SkillClassifier()
        
        # Model instances (lazy loaded)
        self._spacy_model = None
        self._sentence_transformer = None
        self._bert_tokenizer = None
        self._bert_model = None
        self._skill_extraction_pipeline = None
        
        # Thread pool for CPU-intensive tasks
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info("NLP Engine initialized")
    
    @property
    def spacy_model(self):
        """Lazy load spaCy model."""
        if self._spacy_model is None:
            try:
                self._spacy_model = spacy.load("en_core_web_sm")
                logger.info("Loaded spaCy model: en_core_web_sm")
            except OSError:
                logger.warning("spaCy model not found. Using blank model.")
                self._spacy_model = spacy.blank("en")
        return self._spacy_model
    
    @property
    def sentence_transformer(self):
        """Lazy load sentence transformer model."""
        if self._sentence_transformer is None:
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            self._sentence_transformer = SentenceTransformer(
                model_name, 
                cache_folder=str(self.model_cache_dir)
            )
            logger.info(f"Loaded sentence transformer: {model_name}")
        return self._sentence_transformer
    
    @property
    def bert_tokenizer(self):
        """Lazy load BERT tokenizer."""
        if self._bert_tokenizer is None:
            model_name = "microsoft/DialoGPT-medium"
            self._bert_tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=str(self.model_cache_dir)
            )
            logger.info(f"Loaded BERT tokenizer: {model_name}")
        return self._bert_tokenizer
    
    @property
    def bert_model(self):
        """Lazy load BERT model."""
        if self._bert_model is None:
            model_name = "microsoft/DialoGPT-medium"
            self._bert_model = AutoModel.from_pretrained(
                model_name,
                cache_dir=str(self.model_cache_dir)
            )
            logger.info(f"Loaded BERT model: {model_name}")
        return self._bert_model
    
    async def parse_resume(self, file_path: str, file_type: str = None) -> ResumeData:
        """
        Parse resume from PDF or DOC file.
        
        Args:
            file_path: Path to the resume file
            file_type: File type ('pdf', 'docx', 'doc') - auto-detected if None
            
        Returns:
            ResumeData object containing extracted information
        """
        if file_type is None:
            file_type = Path(file_path).suffix.lower().lstrip('.')
        
        try:
            if file_type == 'pdf':
                text = await self._extract_pdf_text(file_path)
            elif file_type in ['docx', 'doc']:
                text = await self._extract_docx_text(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            # Process the extracted text
            resume_data = await self._process_resume_text(text)
            resume_data.file_path = file_path
            resume_data.file_type = file_type
            
            logger.info(f"Successfully parsed resume: {file_path}")
            return resume_data
            
        except Exception as e:
            logger.error(f"Error parsing resume {file_path}: {str(e)}")
            raise
    
    async def _extract_pdf_text(self, file_path: str) -> str:
        """Extract text from PDF using PyMuPDF."""
        def extract():
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, extract)
    
    async def _extract_docx_text(self, file_path: str) -> str:
        """Extract text from DOCX using python-docx."""
        def extract():
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, extract)
    
    async def _process_resume_text(self, text: str) -> ResumeData:
        """
        Process extracted resume text to extract structured information.
        
        Args:
            text: Raw resume text
            
        Returns:
            ResumeData with extracted information
        """
        # Clean and preprocess text
        cleaned_text = self.text_preprocessor.clean_text(text)
        
        # Extract different sections
        sections = self._extract_sections(cleaned_text)
        
        # Extract skills using multiple methods
        skills = await self.extract_skills(cleaned_text)
        
        # Extract other information
        experience = self._extract_experience(sections.get('experience', ''))
        education = self._extract_education(sections.get('education', ''))
        certifications = self._extract_certifications(cleaned_text)
        
        return ResumeData(
            raw_text=text,
            cleaned_text=cleaned_text,
            sections=sections,
            skills=skills,
            experience=experience,
            education=education,
            certifications=certifications
        )
    
    def _extract_sections(self, text: str) -> Dict[str, str]:
        """Extract different sections from resume text."""
        sections = {}
        
        # Common section headers
        section_patterns = {
            'experience': r'(?i)(work\s+experience|professional\s+experience|employment|experience)',
            'education': r'(?i)(education|academic|qualifications)',
            'skills': r'(?i)(skills|technical\s+skills|competencies)',
            'certifications': r'(?i)(certifications?|certificates?)',
            'projects': r'(?i)(projects?|portfolio)',
            'summary': r'(?i)(summary|profile|objective)'
        }
        
        lines = text.split('\n')
        current_section = None
        section_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if line is a section header
            for section_name, pattern in section_patterns.items():
                if re.match(pattern, line):
                    # Save previous section
                    if current_section and section_content:
                        sections[current_section] = '\n'.join(section_content)
                    
                    current_section = section_name
                    section_content = []
                    break
            else:
                # Add line to current section
                if current_section:
                    section_content.append(line)
        
        # Save last section
        if current_section and section_content:
            sections[current_section] = '\n'.join(section_content)
        
        return sections
    
    async def extract_skills(self, text: str) -> List[SkillExtraction]:
        """
        Extract skills from text using multiple NLP techniques.
        
        Args:
            text: Input text to extract skills from
            
        Returns:
            List of SkillExtraction objects with confidence scores
        """
        skills = []
        
        # Method 1: spaCy NER for general entities
        spacy_skills = await self._extract_skills_spacy(text)
        skills.extend(spacy_skills)
        
        # Method 2: Pattern-based extraction for technical skills
        pattern_skills = self._extract_skills_patterns(text)
        skills.extend(pattern_skills)
        
        # Method 3: BERT-based classification
        bert_skills = await self._extract_skills_bert(text)
        skills.extend(bert_skills)
        
        # Merge and deduplicate skills
        merged_skills = self._merge_skills(skills)
        
        # Classify skills into categories
        categorized_skills = await self._categorize_skills(merged_skills)
        
        return categorized_skills
    
    async def _extract_skills_spacy(self, text: str) -> List[SkillExtraction]:
        """Extract skills using spaCy NER."""
        def extract():
            doc = self.spacy_model(text)
            skills = []
            
            for ent in doc.ents:
                if ent.label_ in ['ORG', 'PRODUCT', 'LANGUAGE']:
                    # Check if entity looks like a technical skill
                    if self._is_technical_skill(ent.text):
                        skills.append(SkillExtraction(
                            skill_name=ent.text.strip(),
                            confidence_score=0.7,
                            source='spacy_ner',
                            evidence=[ent.sent.text],
                            category=SkillCategory.TECHNICAL
                        ))
            
            return skills
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, extract)
    
    def _extract_skills_patterns(self, text: str) -> List[SkillExtraction]:
        """Extract skills using regex patterns."""
        skills = []
        
        # Technical skills patterns
        tech_patterns = [
            r'\b(?:Python|Java|JavaScript|TypeScript|C\+\+|C#|Go|Rust|Ruby|PHP|Swift|Kotlin)\b',
            r'\b(?:React|Angular|Vue|Django|Flask|Spring|Express|Laravel)\b',
            r'\b(?:AWS|Azure|GCP|Docker|Kubernetes|Jenkins|Git|Linux|Windows)\b',
            r'\b(?:MySQL|PostgreSQL|MongoDB|Redis|Elasticsearch|Cassandra)\b',
            r'\b(?:TensorFlow|PyTorch|scikit-learn|Pandas|NumPy|OpenCV)\b'
        ]
        
        for pattern in tech_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                skill_name = match.group().strip()
                skills.append(SkillExtraction(
                    skill_name=skill_name,
                    confidence_score=0.8,
                    source='pattern_matching',
                    evidence=[self._get_context(text, match.start(), match.end())],
                    category=SkillCategory.TECHNICAL
                ))
        
        return skills
    
    async def _extract_skills_bert(self, text: str) -> List[SkillExtraction]:
        """Extract skills using BERT-based classification."""
        def extract():
            # Split text into sentences for better processing
            sentences = self.text_preprocessor.split_sentences(text)
            skills = []
            
            for sentence in sentences:
                # Use BERT to identify skill-related sentences
                if self._is_skill_sentence(sentence):
                    # Extract potential skills from the sentence
                    potential_skills = self._extract_noun_phrases(sentence)
                    
                    for skill in potential_skills:
                        if self._is_technical_skill(skill):
                            skills.append(SkillExtraction(
                                skill_name=skill,
                                confidence_score=0.6,
                                source='bert_classification',
                                evidence=[sentence],
                                category=SkillCategory.TECHNICAL
                            ))
            
            return skills
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, extract)
    
    def _is_technical_skill(self, text: str) -> bool:
        """Check if text represents a technical skill."""
        # Simple heuristics for technical skills
        tech_indicators = [
            'programming', 'language', 'framework', 'library', 'database',
            'cloud', 'platform', 'tool', 'technology', 'software'
        ]
        
        text_lower = text.lower()
        
        # Check for common technical terms
        if any(indicator in text_lower for indicator in tech_indicators):
            return True
        
        # Check for typical technical skill patterns
        if re.match(r'^[A-Z][a-z]*(\.[a-z]+)*$', text):  # e.g., React.js
            return True
        
        if re.match(r'^[A-Z]+$', text) and len(text) <= 10:  # e.g., AWS, SQL
            return True
        
        return False
    
    def _is_skill_sentence(self, sentence: str) -> bool:
        """Check if sentence likely contains skill information."""
        skill_indicators = [
            'experience with', 'proficient in', 'skilled in', 'knowledge of',
            'familiar with', 'expertise in', 'worked with', 'using', 'technologies'
        ]
        
        sentence_lower = sentence.lower()
        return any(indicator in sentence_lower for indicator in skill_indicators)
    
    def _extract_noun_phrases(self, text: str) -> List[str]:
        """Extract noun phrases that might be skills."""
        doc = self.spacy_model(text)
        noun_phrases = []
        
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) <= 3:  # Limit to reasonable skill names
                noun_phrases.append(chunk.text.strip())
        
        return noun_phrases
    
    def _get_context(self, text: str, start: int, end: int, context_size: int = 50) -> str:
        """Get context around a match."""
        context_start = max(0, start - context_size)
        context_end = min(len(text), end + context_size)
        return text[context_start:context_end].strip()
    
    def _merge_skills(self, skills: List[SkillExtraction]) -> List[SkillExtraction]:
        """Merge duplicate skills and combine confidence scores."""
        skill_dict = {}
        
        for skill in skills:
            skill_name_lower = skill.skill_name.lower()
            
            if skill_name_lower in skill_dict:
                # Merge with existing skill
                existing = skill_dict[skill_name_lower]
                existing.confidence_score = max(existing.confidence_score, skill.confidence_score)
                existing.evidence.extend(skill.evidence)
                existing.source += f", {skill.source}"
            else:
                skill_dict[skill_name_lower] = skill
        
        return list(skill_dict.values())
    
    async def _categorize_skills(self, skills: List[SkillExtraction]) -> List[SkillExtraction]:
        """Categorize skills using the skill classifier."""
        for skill in skills:
            category = await self.skill_classifier.classify_skill(skill.skill_name)
            skill.category = category
        
        return skills
    
    def _extract_experience(self, experience_text: str) -> List[Dict[str, Any]]:
        """Extract work experience information."""
        experiences = []
        
        # Split by common separators
        entries = re.split(r'\n\s*\n|\n(?=[A-Z][^a-z]*(?:,|\s+\d{4}))', experience_text)
        
        for entry in entries:
            if not entry.strip():
                continue
            
            experience = self._parse_experience_entry(entry)
            if experience:
                experiences.append(experience)
        
        return experiences
    
    def _parse_experience_entry(self, entry: str) -> Optional[Dict[str, Any]]:
        """Parse a single experience entry."""
        lines = [line.strip() for line in entry.split('\n') if line.strip()]
        if not lines:
            return None
        
        # First line usually contains position and company
        first_line = lines[0]
        
        # Extract dates
        date_pattern = r'(\d{4})\s*[-–]\s*(\d{4}|present|current)'
        date_match = re.search(date_pattern, entry, re.IGNORECASE)
        
        return {
            'raw_text': entry,
            'first_line': first_line,
            'dates': date_match.group() if date_match else None,
            'description': '\n'.join(lines[1:]) if len(lines) > 1 else ''
        }
    
    def _extract_education(self, education_text: str) -> List[Dict[str, Any]]:
        """Extract education information."""
        education_entries = []
        
        # Split by common separators
        entries = re.split(r'\n\s*\n|\n(?=[A-Z][^a-z]*(?:,|\s+\d{4}))', education_text)
        
        for entry in entries:
            if not entry.strip():
                continue
            
            education = self._parse_education_entry(entry)
            if education:
                education_entries.append(education)
        
        return education_entries
    
    def _parse_education_entry(self, entry: str) -> Optional[Dict[str, Any]]:
        """Parse a single education entry."""
        lines = [line.strip() for line in entry.split('\n') if line.strip()]
        if not lines:
            return None
        
        # Extract degree and institution
        first_line = lines[0]
        
        # Extract graduation year
        year_pattern = r'\b(19|20)\d{2}\b'
        year_match = re.search(year_pattern, entry)
        
        return {
            'raw_text': entry,
            'degree_institution': first_line,
            'graduation_year': year_match.group() if year_match else None,
            'details': '\n'.join(lines[1:]) if len(lines) > 1 else ''
        }
    
    def _extract_certifications(self, text: str) -> List[str]:
        """Extract certifications from text."""
        certifications = []
        
        # Common certification patterns
        cert_patterns = [
            r'(?i)certified?\s+[a-z\s]+(?:professional|specialist|expert|associate)',
            r'(?i)[a-z\s]+\s+certification',
            r'(?i)(?:AWS|Azure|GCP|Google|Microsoft|Oracle|Cisco|CompTIA)\s+[A-Z\-\d]+',
        ]
        
        for pattern in cert_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                cert = match.group().strip()
                if len(cert) > 5:  # Filter out very short matches
                    certifications.append(cert)
        
        return list(set(certifications))  # Remove duplicates
    
    async def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate semantic embeddings for texts using sentence transformers.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Numpy array of embeddings
        """
        def embed():
            return self.sentence_transformer.encode(texts)
        
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(self.executor, embed)
        
        logger.info(f"Generated embeddings for {len(texts)} texts")
        return embeddings
    
    async def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        embeddings = await self.generate_embeddings([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(similarity)
    
    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
        logger.info("NLP Engine cleaned up")