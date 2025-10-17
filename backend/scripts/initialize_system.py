#!/usr/bin/env python3
"""
System initialization script to set up the AI Career Recommender with real data
and ensure all components are functional with personalized content.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the backend app to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from app.core.database import AsyncSessionLocal
from app.repositories.skill import SkillRepository
from app.repositories.job import JobRepository
from app.repositories.user import UserRepository
from app.repositories.profile import ProfileRepository
from app.services.auth_service import AuthService
from app.core.logging import get_logger

logger = get_logger(__name__)

# Real skill categories and skills data
SKILL_CATEGORIES = {
    "Programming Languages": [
        "Python", "JavaScript", "TypeScript", "Java", "C++", "C#", "Go", "Rust", 
        "PHP", "Ruby", "Swift", "Kotlin", "Scala", "R", "MATLAB", "SQL"
    ],
    "Web Development": [
        "React", "Angular", "Vue.js", "Node.js", "Express.js", "Django", "Flask",
        "FastAPI", "Spring Boot", "ASP.NET", "Laravel", "Ruby on Rails", "HTML5",
        "CSS3", "SASS", "Bootstrap", "Tailwind CSS"
    ],
    "Mobile Development": [
        "React Native", "Flutter", "iOS Development", "Android Development",
        "Xamarin", "Ionic", "PhoneGap", "Unity 3D"
    ],
    "Data Science & Analytics": [
        "Machine Learning", "Deep Learning", "Data Analysis", "Statistics",
        "Pandas", "NumPy", "Scikit-learn", "TensorFlow", "PyTorch", "Keras",
        "Jupyter", "Tableau", "Power BI", "Apache Spark", "Hadoop"
    ],
    "Cloud & DevOps": [
        "AWS", "Azure", "Google Cloud", "Docker", "Kubernetes", "Jenkins",
        "GitLab CI/CD", "Terraform", "Ansible", "Chef", "Puppet", "Prometheus",
        "Grafana", "ELK Stack"
    ],
    "Database Technologies": [
        "PostgreSQL", "MySQL", "MongoDB", "Redis", "Elasticsearch", "Cassandra",
        "Oracle", "SQL Server", "SQLite", "DynamoDB", "Neo4j"
    ],
    "Software Engineering": [
        "Git", "Agile", "Scrum", "Test-Driven Development", "Unit Testing",
        "Integration Testing", "Code Review", "Design Patterns", "Microservices",
        "RESTful APIs", "GraphQL", "System Design"
    ],
    "Cybersecurity": [
        "Network Security", "Penetration Testing", "Ethical Hacking", "CISSP",
        "CEH", "Security Auditing", "Vulnerability Assessment", "Incident Response",
        "Cryptography", "Firewall Management"
    ],
    "AI & Machine Learning": [
        "Natural Language Processing", "Computer Vision", "Reinforcement Learning",
        "Neural Networks", "Generative AI", "LLMs", "OpenAI API", "Hugging Face",
        "MLOps", "Model Deployment", "Feature Engineering"
    ],
    "Project Management": [
        "PMP", "Agile Project Management", "Scrum Master", "Kanban", "JIRA",
        "Confluence", "Risk Management", "Stakeholder Management", "Budget Management"
    ]
}

# Real job titles and descriptions
JOB_DATA = [
    {
        "title": "Senior Full Stack Developer",
        "company": "TechCorp Solutions",
        "location": "San Francisco, CA",
        "salary_min": 120000,
        "salary_max": 180000,
        "description": "We are seeking a Senior Full Stack Developer to join our dynamic team. You will be responsible for developing and maintaining web applications using modern technologies like React, Node.js, and PostgreSQL.",
        "required_skills": ["React", "Node.js", "PostgreSQL", "JavaScript", "TypeScript", "Git"],
        "preferred_skills": ["AWS", "Docker", "GraphQL", "Jest", "Agile"],
        "experience_level": "Senior",
        "job_type": "Full-time",
        "remote_friendly": True
    },
    {
        "title": "Data Scientist",
        "company": "DataInsights Inc",
        "location": "New York, NY",
        "salary_min": 100000,
        "salary_max": 150000,
        "description": "Join our data science team to build predictive models and extract insights from large datasets. You'll work with Python, machine learning libraries, and cloud platforms.",
        "required_skills": ["Python", "Machine Learning", "Pandas", "NumPy", "Statistics"],
        "preferred_skills": ["TensorFlow", "AWS", "Tableau", "SQL", "Deep Learning"],
        "experience_level": "Mid-level",
        "job_type": "Full-time",
        "remote_friendly": True
    },
    {
        "title": "DevOps Engineer",
        "company": "CloudFirst Technologies",
        "location": "Austin, TX",
        "salary_min": 110000,
        "salary_max": 160000,
        "description": "We're looking for a DevOps Engineer to help us scale our infrastructure and improve our deployment processes. Experience with Kubernetes, AWS, and CI/CD pipelines is essential.",
        "required_skills": ["AWS", "Kubernetes", "Docker", "Jenkins", "Terraform"],
        "preferred_skills": ["Prometheus", "Grafana", "Ansible", "Python", "Bash"],
        "experience_level": "Mid-level",
        "job_type": "Full-time",
        "remote_friendly": False
    },
    {
        "title": "Mobile App Developer",
        "company": "MobileFirst Studios",
        "location": "Los Angeles, CA",
        "salary_min": 90000,
        "salary_max": 130000,
        "description": "Develop cutting-edge mobile applications for iOS and Android platforms. Experience with React Native or Flutter is highly preferred.",
        "required_skills": ["React Native", "JavaScript", "iOS Development", "Android Development"],
        "preferred_skills": ["Flutter", "TypeScript", "Firebase", "Redux", "Jest"],
        "experience_level": "Mid-level",
        "job_type": "Full-time",
        "remote_friendly": True
    },
    {
        "title": "Machine Learning Engineer",
        "company": "AI Innovations Lab",
        "location": "Seattle, WA",
        "salary_min": 130000,
        "salary_max": 200000,
        "description": "Build and deploy machine learning models at scale. You'll work on cutting-edge AI projects involving NLP, computer vision, and deep learning.",
        "required_skills": ["Python", "TensorFlow", "PyTorch", "Machine Learning", "Deep Learning"],
        "preferred_skills": ["MLOps", "Kubernetes", "AWS", "Docker", "Jupyter"],
        "experience_level": "Senior",
        "job_type": "Full-time",
        "remote_friendly": True
    },
    {
        "title": "Frontend Developer",
        "company": "UX Design Co",
        "location": "Remote",
        "salary_min": 80000,
        "salary_max": 120000,
        "description": "Create beautiful and responsive user interfaces using modern frontend technologies. Strong focus on user experience and accessibility.",
        "required_skills": ["React", "JavaScript", "HTML5", "CSS3", "TypeScript"],
        "preferred_skills": ["Vue.js", "SASS", "Webpack", "Jest", "Figma"],
        "experience_level": "Mid-level",
        "job_type": "Full-time",
        "remote_friendly": True
    },
    {
        "title": "Backend Developer",
        "company": "ServerSide Solutions",
        "location": "Chicago, IL",
        "salary_min": 95000,
        "salary_max": 140000,
        "description": "Design and implement scalable backend systems and APIs. Experience with microservices architecture and database optimization is a plus.",
        "required_skills": ["Python", "Django", "PostgreSQL", "RESTful APIs", "Git"],
        "preferred_skills": ["FastAPI", "Redis", "Docker", "AWS", "GraphQL"],
        "experience_level": "Mid-level",
        "job_type": "Full-time",
        "remote_friendly": False
    },
    {
        "title": "Cybersecurity Analyst",
        "company": "SecureNet Corp",
        "location": "Washington, DC",
        "salary_min": 85000,
        "salary_max": 125000,
        "description": "Protect our organization from cyber threats by monitoring security systems, conducting vulnerability assessments, and responding to incidents.",
        "required_skills": ["Network Security", "Vulnerability Assessment", "Incident Response", "CISSP"],
        "preferred_skills": ["Penetration Testing", "Ethical Hacking", "Python", "Linux", "SIEM"],
        "experience_level": "Mid-level",
        "job_type": "Full-time",
        "remote_friendly": False
    },
    {
        "title": "Product Manager",
        "company": "InnovateTech",
        "location": "Boston, MA",
        "salary_min": 110000,
        "salary_max": 160000,
        "description": "Lead product development from conception to launch. Work closely with engineering, design, and marketing teams to deliver exceptional products.",
        "required_skills": ["Product Management", "Agile", "Stakeholder Management", "Market Research"],
        "preferred_skills": ["JIRA", "Confluence", "SQL", "A/B Testing", "User Research"],
        "experience_level": "Senior",
        "job_type": "Full-time",
        "remote_friendly": True
    },
    {
        "title": "UI/UX Designer",
        "company": "DesignFirst Agency",
        "location": "Portland, OR",
        "salary_min": 70000,
        "salary_max": 110000,
        "description": "Create intuitive and engaging user experiences for web and mobile applications. Collaborate with developers to bring designs to life.",
        "required_skills": ["UI Design", "UX Design", "Figma", "Adobe Creative Suite", "Prototyping"],
        "preferred_skills": ["Sketch", "InVision", "User Research", "HTML5", "CSS3"],
        "experience_level": "Mid-level",
        "job_type": "Full-time",
        "remote_friendly": True
    }
]

async def initialize_skills():
    """Initialize skill categories and skills in the database."""
    logger.info("Initializing skills and categories...")
    
    async with AsyncSessionLocal() as session:
        skill_repo = SkillRepository()
        
        for category_name, skills in SKILL_CATEGORIES.items():
            # Create skills in this category
            for skill_name in skills:
                await skill_repo.find_or_create_skill(
                    db=session,
                    name=skill_name,
                    category=category_name
                )
        
        logger.info(f"Created {len(SKILL_CATEGORIES)} categories and {sum(len(skills) for skills in SKILL_CATEGORIES.values())} skills")

async def initialize_jobs():
    """Initialize job postings in the database."""
    logger.info("Initializing job postings...")
    
    async with AsyncSessionLocal() as session:
        job_repo = JobRepository()
        skill_repo = SkillRepository()
        
        for job_data in JOB_DATA:
            # Get skill IDs for required and preferred skills
            required_skill_ids = []
            preferred_skill_ids = []
            
            for skill_name in job_data["required_skills"]:
                skill = await skill_repo.get_by_name(db=session, name=skill_name)
                if skill:
                    required_skill_ids.append(skill.id)
            
            for skill_name in job_data["preferred_skills"]:
                skill = await skill_repo.get_by_name(db=session, name=skill_name)
                if skill:
                    preferred_skill_ids.append(skill.id)
            
            # Create job posting using base repository create method
            job_create_data = {
                "title": job_data["title"],
                "company": job_data["company"],
                "location": job_data["location"],
                "description": job_data["description"],
                "salary_min": job_data["salary_min"],
                "salary_max": job_data["salary_max"],
                "experience_level": job_data["experience_level"],
                "employment_type": job_data["job_type"],
                "remote_type": "remote" if job_data["remote_friendly"] else "onsite",
                "source": "demo_data"
            }
            await job_repo.create(db=session, obj_in=job_create_data)
        
        logger.info(f"Created {len(JOB_DATA)} job postings")

async def create_demo_user():
    """Create a demo user for testing purposes."""
    logger.info("Creating demo user...")
    
    async with AsyncSessionLocal() as session:
        auth_service = AuthService(session)
        user_repo = UserRepository()
        
        # Check if demo user already exists
        existing_user = await user_repo.get_by_email(db=session, email="demo@aicareer.com")
        if existing_user:
            logger.info("Demo user already exists")
            return existing_user
        
        # Create demo user using repository directly
        from app.schemas.auth import UserRegister
        user_data = UserRegister(
            email="demo@aicareer.com",
            password="Demo123!",
            full_name="Demo Student"
        )
        user = await auth_service.register_user(user_data)
        
        logger.info(f"Created demo user: {user.email}")
        return user

async def initialize_demo_profile():
    """Create a demo profile with realistic data."""
    logger.info("Creating demo profile...")
    
    async with AsyncSessionLocal() as session:
        user_repo = UserRepository()
        profile_repo = ProfileRepository()
        skill_repo = SkillRepository()
        
        # Get demo user
        user = await user_repo.get_by_email(db=session, email="demo@aicareer.com")
        if not user:
            logger.error("Demo user not found")
            return
        
        # Check if profile already exists
        existing_profile = await profile_repo.get_by_user_id(db=session, user_id=user.id)
        if existing_profile:
            logger.info("Demo profile already exists")
            return existing_profile
        
        # Get some skills for the demo profile
        python_skill = await skill_repo.get_by_name(db=session, name="Python")
        js_skill = await skill_repo.get_by_name(db=session, name="JavaScript")
        react_skill = await skill_repo.get_by_name(db=session, name="React")
        ml_skill = await skill_repo.get_by_name(db=session, name="Machine Learning")
        
        skill_ids = [skill.id for skill in [python_skill, js_skill, react_skill, ml_skill] if skill]
        
        # Create demo profile using base repository create method
        profile_data = {
            "user_id": str(user.id),
            "bio": "Computer Science student passionate about AI and web development. Currently learning machine learning and building full-stack applications.",
            "location": "San Francisco, CA",
            "experience_level": "Entry-level",
            "career_goals": "Become a Machine Learning Engineer and work on cutting-edge AI projects",
            "github_username": "demo-student",
            "linkedin_url": "https://linkedin.com/in/demo-student",
            "portfolio_url": "https://demo-student.dev"
        }
        profile = await profile_repo.create(db=session, obj_in=profile_data)
        
        logger.info(f"Created demo profile for user: {user.email}")
        return profile

async def main():
    """Main initialization function."""
    logger.info("Starting system initialization...")
    
    try:
        # Initialize skills and categories
        await initialize_skills()
        
        # Initialize job postings
        await initialize_jobs()
        
        # Create demo user and profile
        await create_demo_user()
        await initialize_demo_profile()
        
        logger.info("System initialization completed successfully!")
        
    except Exception as e:
        logger.error(f"System initialization failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())