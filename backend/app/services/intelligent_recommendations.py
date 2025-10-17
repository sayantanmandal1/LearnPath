"""
Simple intelligent career recommendations system
"""
from datetime import datetime

def generate_career_recommendations(job_title="", skills=None):
    """Generate recommendations for any job profile"""
    if skills is None:
        skills = []
    
    job_title_lower = job_title.lower()
    
    # Comprehensive job profile matching
    if any(keyword in job_title_lower for keyword in ["backend", "server", "api", "database"]):
        return get_backend_recommendations()
    elif any(keyword in job_title_lower for keyword in ["frontend", "react", "javascript", "ui"]):
        return get_frontend_recommendations()
    elif any(keyword in job_title_lower for keyword in ["marketing", "digital", "social", "content"]):
        return get_marketing_recommendations()
    elif any(keyword in job_title_lower for keyword in ["data", "analyst", "science", "machine learning"]):
        return get_data_recommendations()
    else:
        return get_default_recommendations()

def get_backend_recommendations():
    return [
        {
            "job_title": "Senior Backend Developer",
            "company": "TechStart Inc",
            "match_score": 0.92,
            "required_skills": ["SQL", "Python", "Node.js", "PostgreSQL", "Docker"],
            "skill_gaps": {"microservices": 0.2, "kubernetes": 0.3},
            "salary_range": "$85,000 - $130,000",
            "growth_potential": "90%",
            "market_demand": "Very High",
            "reasoning": "Perfect match for backend developer with SQL expertise",
            "alternative_paths": ["Full Stack Developer", "Database Engineer"],
            "location": "Remote",
            "employment_type": "Full-time",
            "confidence_score": 0.95,
            "recommendation_date": datetime.now().isoformat()
        },
        {
            "job_title": "Database Engineer",
            "company": "DataSolutions Corp", 
            "match_score": 0.88,
            "required_skills": ["SQL", "PostgreSQL", "MySQL", "Performance Tuning"],
            "skill_gaps": {"nosql": 0.4, "data_warehousing": 0.3},
            "salary_range": "$75,000 - $115,000",
            "growth_potential": "80%",
            "market_demand": "High",
            "reasoning": "Strong SQL skills make you ideal for database specialization",
            "alternative_paths": ["Data Engineer", "Backend Developer"],
            "location": "New York, NY",
            "employment_type": "Full-time",
            "confidence_score": 0.9,
            "recommendation_date": datetime.now().isoformat()
        }
    ]

def get_frontend_recommendations():
    return [
        {
            "job_title": "Senior Frontend Developer",
            "company": "UITech Solutions",
            "match_score": 0.90,
            "required_skills": ["React", "JavaScript", "TypeScript", "CSS", "HTML"],
            "skill_gaps": {"next.js": 0.3, "testing": 0.2},
            "salary_range": "$75,000 - $125,000",
            "growth_potential": "85%",
            "market_demand": "Very High",
            "reasoning": "Strong frontend skills with modern framework experience",
            "alternative_paths": ["Full Stack Developer", "UI/UX Developer"],
            "location": "San Francisco, CA",
            "employment_type": "Full-time",
            "confidence_score": 0.9,
            "recommendation_date": datetime.now().isoformat()
        }
    ]

def get_marketing_recommendations():
    return [
        {
            "job_title": "Digital Marketing Manager",
            "company": "GrowthTech Marketing",
            "match_score": 0.88,
            "required_skills": ["SEO", "Google Analytics", "Social Media", "Content Strategy"],
            "skill_gaps": {"paid_advertising": 0.3, "automation": 0.4},
            "salary_range": "$55,000 - $95,000",
            "growth_potential": "90%",
            "market_demand": "Very High",
            "reasoning": "Strong digital marketing foundation with growth opportunities",
            "alternative_paths": ["Content Manager", "Growth Hacker"],
            "location": "Austin, TX",
            "employment_type": "Full-time",
            "confidence_score": 0.88,
            "recommendation_date": datetime.now().isoformat()
        }
    ]

def get_data_recommendations():
    return [
        {
            "job_title": "Data Scientist",
            "company": "Analytics Corp",
            "match_score": 0.93,
            "required_skills": ["Python", "SQL", "Machine Learning", "Statistics"],
            "skill_gaps": {"deep_learning": 0.4, "big_data": 0.3},
            "salary_range": "$90,000 - $150,000",
            "growth_potential": "95%",
            "market_demand": "Extremely High",
            "reasoning": "Data science is one of the fastest growing fields",
            "alternative_paths": ["ML Engineer", "Business Analyst"],
            "location": "Seattle, WA",
            "employment_type": "Full-time",
            "confidence_score": 0.93,
            "recommendation_date": datetime.now().isoformat()
        }
    ]

def get_default_recommendations():
    return [
        {
            "job_title": "Technology Consultant",
            "company": "TechConsulting Pro",
            "match_score": 0.75,
            "required_skills": ["Problem Solving", "Communication", "Technology", "Analysis"],
            "skill_gaps": {"industry_knowledge": 0.3, "technical_skills": 0.4},
            "salary_range": "$60,000 - $100,000",
            "growth_potential": "80%",
            "market_demand": "Medium",
            "reasoning": "Versatile role that can leverage diverse skill sets",
            "alternative_paths": ["Project Manager", "Business Analyst"],
            "location": "Remote",
            "employment_type": "Full-time",
            "confidence_score": 0.75,
            "recommendation_date": datetime.now().isoformat()
        }
    ]