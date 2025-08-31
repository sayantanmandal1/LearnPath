"""
Test data generators and utilities for comprehensive testing.
"""
import random
import json
import tempfile
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from faker import Faker
import uuid


class TestDataGenerator:
    """Generate realistic test data for various testing scenarios."""
    
    def __init__(self, seed: int = 42):
        """Initialize with a seed for reproducible data generation."""
        self.fake = Faker()
        Faker.seed(seed)
        random.seed(seed)
        
        # Predefined skill categories and skills
        self.skill_categories = {
            "programming": [
                "Python", "JavaScript", "Java", "C++", "Go", "Rust", "TypeScript",
                "C#", "PHP", "Ruby", "Swift", "Kotlin", "Scala", "R"
            ],
            "frameworks": [
                "React", "Angular", "Vue.js", "Django", "Flask", "FastAPI", "Express.js",
                "Spring Boot", "Laravel", "Ruby on Rails", "ASP.NET", "Next.js"
            ],
            "databases": [
                "PostgreSQL", "MySQL", "MongoDB", "Redis", "Elasticsearch", "SQLite",
                "Oracle", "Cassandra", "DynamoDB", "Neo4j"
            ],
            "cloud": [
                "AWS", "Azure", "Google Cloud", "Docker", "Kubernetes", "Terraform",
                "Jenkins", "GitLab CI", "GitHub Actions", "Ansible"
            ],
            "ml_ai": [
                "Machine Learning", "Deep Learning", "TensorFlow", "PyTorch", "scikit-learn",
                "Pandas", "NumPy", "Computer Vision", "NLP", "Neural Networks"
            ],
            "soft_skills": [
                "Leadership", "Communication", "Problem Solving", "Team Collaboration",
                "Project Management", "Agile", "Scrum", "Mentoring", "Public Speaking"
            ],
            "tools": [
                "Git", "VS Code", "IntelliJ", "Postman", "Jira", "Confluence",
                "Slack", "Figma", "Adobe Creative Suite", "Tableau"
            ]
        }
        
        self.job_titles = [
            "Software Engineer", "Senior Software Engineer", "Lead Software Engineer",
            "Frontend Developer", "Backend Developer", "Full Stack Developer",
            "Data Scientist", "ML Engineer", "DevOps Engineer", "Product Manager",
            "Tech Lead", "Engineering Manager", "Principal Engineer", "Architect",
            "QA Engineer", "Security Engineer", "Mobile Developer", "Game Developer"
        ]
        
        self.companies = [
            "Google", "Microsoft", "Amazon", "Apple", "Meta", "Netflix", "Tesla",
            "Uber", "Airbnb", "Spotify", "Stripe", "Shopify", "Slack", "Zoom",
            "TechCorp", "InnovateLabs", "DataDriven Inc", "CloudFirst Solutions"
        ]
        
        self.locations = [
            "San Francisco, CA", "New York, NY", "Seattle, WA", "Austin, TX",
            "Boston, MA", "Los Angeles, CA", "Chicago, IL", "Denver, CO",
            "Remote", "London, UK", "Berlin, Germany", "Toronto, Canada"
        ]
    
    def generate_user_data(self, count: int = 1) -> List[Dict[str, Any]]:
        """Generate realistic user registration data."""
        users = []
        
        for _ in range(count):
            user = {
                "email": self.fake.email(),
                "password": self.fake.password(length=12),
                "full_name": self.fake.name(),
                "created_at": self.fake.date_time_between(start_date="-2y", end_date="now")
            }
            users.append(user)
        
        return users
    
    def generate_profile_data(self, count: int = 1) -> List[Dict[str, Any]]:
        """Generate realistic user profile data."""
        profiles = []
        
        for _ in range(count):
            # Select random skills from different categories
            selected_skills = []
            for category, skills in self.skill_categories.items():
                num_skills = random.randint(1, min(4, len(skills)))
                selected_skills.extend(random.sample(skills, num_skills))
            
            # Remove duplicates and limit total skills
            selected_skills = list(set(selected_skills))[:random.randint(5, 15)]
            
            profile = {
                "skills": selected_skills,
                "dream_job": random.choice(self.job_titles),
                "experience_years": random.randint(0, 15),
                "github_username": self.fake.user_name() if random.random() > 0.3 else None,
                "leetcode_id": self.fake.user_name() if random.random() > 0.5 else None,
                "linkedin_url": f"https://linkedin.com/in/{self.fake.user_name()}" if random.random() > 0.2 else None,
                "location": random.choice(self.locations),
                "bio": self.fake.text(max_nb_chars=200)
            }
            profiles.append(profile)
        
        return profiles
    
    def generate_resume_text(self, profile_data: Dict[str, Any]) -> str:
        """Generate realistic resume text based on profile data."""
        name = self.fake.name()
        email = self.fake.email()
        phone = self.fake.phone_number()
        
        resume_parts = [
            f"{name}",
            f"Email: {email} | Phone: {phone}",
            f"Location: {profile_data.get('location', 'Remote')}",
            "",
            "PROFESSIONAL SUMMARY",
            f"Experienced {profile_data['dream_job'].lower()} with {profile_data['experience_years']} years of experience.",
            f"Skilled in {', '.join(profile_data['skills'][:5])} and passionate about technology innovation.",
            "",
            "TECHNICAL SKILLS",
        ]
        
        # Group skills by category
        for category, skills in self.skill_categories.items():
            profile_skills_in_category = [s for s in profile_data['skills'] if s in skills]
            if profile_skills_in_category:
                resume_parts.append(f"{category.replace('_', ' ').title()}: {', '.join(profile_skills_in_category)}")
        
        resume_parts.extend([
            "",
            "WORK EXPERIENCE",
        ])
        
        # Generate work experience
        years_left = profile_data['experience_years']
        while years_left > 0:
            job_duration = min(random.randint(1, 4), years_left)
            company = random.choice(self.companies)
            title = random.choice(self.job_titles)
            
            resume_parts.extend([
                f"{title} | {company} | {job_duration} year{'s' if job_duration > 1 else ''}",
                f"• Developed applications using {random.choice(profile_data['skills'])}",
                f"• Collaborated with cross-functional teams on {random.choice(['web applications', 'mobile apps', 'data pipelines', 'ML models'])}",
                f"• Improved system performance by {random.randint(10, 50)}%",
                ""
            ])
            
            years_left -= job_duration
        
        resume_parts.extend([
            "EDUCATION",
            f"Bachelor of Science in {random.choice(['Computer Science', 'Software Engineering', 'Data Science'])}",
            f"{self.fake.company()} University",
            ""
        ])
        
        return "\n".join(resume_parts)
    
    def generate_job_postings(self, count: int = 50) -> List[Dict[str, Any]]:
        """Generate realistic job posting data."""
        jobs = []
        
        for _ in range(count):
            title = random.choice(self.job_titles)
            company = random.choice(self.companies)
            location = random.choice(self.locations)
            
            # Select required skills based on job title
            required_skills = []
            if "Frontend" in title or "React" in title:
                required_skills.extend(random.sample(self.skill_categories["programming"][:3], 2))
                required_skills.extend(random.sample(self.skill_categories["frameworks"][:6], 2))
            elif "Backend" in title or "API" in title:
                required_skills.extend(random.sample(self.skill_categories["programming"], 2))
                required_skills.extend(random.sample(self.skill_categories["databases"], 2))
            elif "Data" in title or "ML" in title:
                required_skills.extend(["Python", "SQL"])
                required_skills.extend(random.sample(self.skill_categories["ml_ai"], 3))
            elif "DevOps" in title:
                required_skills.extend(random.sample(self.skill_categories["cloud"], 4))
            else:
                # General software engineering role
                required_skills.extend(random.sample(self.skill_categories["programming"], 2))
                required_skills.extend(random.sample(self.skill_categories["frameworks"], 1))
            
            # Add some soft skills
            required_skills.extend(random.sample(self.skill_categories["soft_skills"], 2))
            
            # Determine experience level and salary
            if "Senior" in title or "Lead" in title:
                experience_level = "senior"
                salary_min = random.randint(120000, 150000)
                salary_max = salary_min + random.randint(30000, 50000)
            elif "Principal" in title or "Manager" in title:
                experience_level = "senior"
                salary_min = random.randint(150000, 200000)
                salary_max = salary_min + random.randint(50000, 100000)
            elif "Junior" in title or random.random() < 0.2:
                experience_level = "junior"
                salary_min = random.randint(60000, 80000)
                salary_max = salary_min + random.randint(15000, 25000)
            else:
                experience_level = "mid"
                salary_min = random.randint(90000, 120000)
                salary_max = salary_min + random.randint(20000, 40000)
            
            job = {
                "title": title,
                "company": company,
                "location": location,
                "description": f"We are looking for a talented {title.lower()} to join our team at {company}. "
                             f"The ideal candidate will have experience with {', '.join(required_skills[:3])} "
                             f"and a passion for building high-quality software solutions.",
                "required_skills": list(set(required_skills)),
                "experience_level": experience_level,
                "salary_min": salary_min,
                "salary_max": salary_max,
                "source": random.choice(["linkedin", "indeed", "glassdoor", "company_website"]),
                "posted_date": self.fake.date_time_between(start_date="-30d", end_date="now"),
                "remote_friendly": random.random() > 0.4,
                "benefits": random.sample([
                    "Health Insurance", "401k Matching", "Flexible PTO", "Remote Work",
                    "Stock Options", "Learning Budget", "Gym Membership", "Free Lunch"
                ], random.randint(3, 6))
            }
            jobs.append(job)
        
        return jobs
    
    def generate_github_profile_data(self, username: str) -> Dict[str, Any]:
        """Generate realistic GitHub profile data."""
        languages = random.sample(self.skill_categories["programming"], random.randint(2, 5))
        language_percentages = {}
        remaining = 100
        
        for i, lang in enumerate(languages):
            if i == len(languages) - 1:
                language_percentages[lang] = remaining
            else:
                percentage = random.randint(10, remaining - (len(languages) - i - 1) * 10)
                language_percentages[lang] = percentage
                remaining -= percentage
        
        repositories = []
        for _ in range(random.randint(5, 20)):
            repo_name = f"{random.choice(['awesome', 'super', 'cool', 'my'])}-{random.choice(['project', 'app', 'tool', 'lib'])}"
            repositories.append({
                "name": repo_name,
                "language": random.choice(languages),
                "stars": random.randint(0, 100),
                "forks": random.randint(0, 20),
                "description": self.fake.sentence(),
                "updated_at": self.fake.date_time_between(start_date="-1y", end_date="now")
            })
        
        return {
            "username": username,
            "name": self.fake.name(),
            "bio": self.fake.text(max_nb_chars=100),
            "location": random.choice(self.locations),
            "company": random.choice(self.companies) if random.random() > 0.3 else None,
            "repositories": repositories,
            "languages": language_percentages,
            "total_commits": random.randint(100, 2000),
            "followers": random.randint(5, 500),
            "following": random.randint(10, 200),
            "public_repos": len(repositories),
            "created_at": self.fake.date_time_between(start_date="-5y", end_date="-1y")
        }
    
    def generate_leetcode_profile_data(self, username: str) -> Dict[str, Any]:
        """Generate realistic LeetCode profile data."""
        total_problems = random.randint(50, 500)
        easy_solved = random.randint(int(total_problems * 0.4), int(total_problems * 0.6))
        medium_solved = random.randint(int(total_problems * 0.3), int(total_problems * 0.5))
        hard_solved = total_problems - easy_solved - medium_solved
        
        if hard_solved < 0:
            hard_solved = random.randint(0, int(total_problems * 0.1))
            medium_solved = total_problems - easy_solved - hard_solved
        
        algorithm_skills = [
            "Dynamic Programming", "Graph Theory", "Binary Search", "Two Pointers",
            "Sliding Window", "Backtracking", "Greedy", "Divide and Conquer",
            "Tree Traversal", "Hash Table", "Stack", "Queue", "Heap"
        ]
        
        return {
            "username": username,
            "problems_solved": total_problems,
            "easy_solved": easy_solved,
            "medium_solved": medium_solved,
            "hard_solved": hard_solved,
            "acceptance_rate": round(random.uniform(0.3, 0.8), 2),
            "contest_rating": random.randint(1200, 2000) if random.random() > 0.3 else None,
            "contest_attended": random.randint(0, 50),
            "skills": random.sample(algorithm_skills, random.randint(3, 8)),
            "badges": random.sample([
                "50 Days Badge", "100 Days Badge", "Annual Badge 2023",
                "Contest Badge", "Study Plan Badge"
            ], random.randint(1, 3)),
            "streak": random.randint(0, 100)
        }
    
    def generate_linkedin_profile_data(self, name: str) -> Dict[str, Any]:
        """Generate realistic LinkedIn profile data."""
        experience = []
        total_years = random.randint(1, 15)
        years_covered = 0
        
        while years_covered < total_years:
            duration_years = min(random.randint(1, 4), total_years - years_covered)
            
            experience.append({
                "title": random.choice(self.job_titles),
                "company": random.choice(self.companies),
                "duration": f"{duration_years} year{'s' if duration_years > 1 else ''}",
                "location": random.choice(self.locations),
                "description": self.fake.text(max_nb_chars=200),
                "skills": random.sample([skill for skills in self.skill_categories.values() for skill in skills], 
                                      random.randint(3, 8))
            })
            years_covered += duration_years
        
        return {
            "name": name,
            "headline": f"{random.choice(self.job_titles)} at {random.choice(self.companies)}",
            "location": random.choice(self.locations),
            "summary": self.fake.text(max_nb_chars=300),
            "experience": experience,
            "education": [
                {
                    "school": f"{self.fake.company()} University",
                    "degree": f"Bachelor of Science in {random.choice(['Computer Science', 'Software Engineering', 'Information Technology'])}",
                    "field_of_study": random.choice(["Computer Science", "Software Engineering", "Data Science"]),
                    "years": f"{random.randint(2010, 2020)}-{random.randint(2014, 2024)}"
                }
            ],
            "skills": random.sample([skill for skills in self.skill_categories.values() for skill in skills], 
                                  random.randint(10, 20)),
            "connections": random.randint(50, 1000),
            "endorsements": {
                skill: random.randint(1, 20) 
                for skill in random.sample([skill for skills in self.skill_categories.values() for skill in skills], 5)
            }
        }
    
    def generate_learning_resources(self, skills: List[str], count: int = 10) -> List[Dict[str, Any]]:
        """Generate realistic learning resource data."""
        providers = ["Coursera", "Udemy", "edX", "Pluralsight", "LinkedIn Learning", "YouTube", "freeCodeCamp"]
        resource_types = ["course", "tutorial", "book", "project", "bootcamp", "certification"]
        
        resources = []
        
        for _ in range(count):
            skill = random.choice(skills)
            resource_type = random.choice(resource_types)
            provider = random.choice(providers)
            
            # Generate realistic titles
            title_templates = [
                f"Complete {skill} {resource_type.title()}",
                f"Master {skill} in 30 Days",
                f"{skill} for Beginners",
                f"Advanced {skill} Techniques",
                f"Build Projects with {skill}",
                f"{skill} Certification Prep"
            ]
            
            title = random.choice(title_templates)
            
            # Determine cost based on provider
            if provider in ["YouTube", "freeCodeCamp"]:
                cost = 0
            elif provider == "Coursera":
                cost = random.choice([0, 39, 49, 79])  # Some free, some paid
            else:
                cost = random.randint(20, 200)
            
            resource = {
                "title": title,
                "type": resource_type,
                "provider": provider,
                "skill": skill,
                "duration_hours": random.randint(5, 100),
                "rating": round(random.uniform(3.5, 5.0), 1),
                "num_reviews": random.randint(100, 10000),
                "cost": cost,
                "difficulty": random.choice(["beginner", "intermediate", "advanced"]),
                "url": f"https://{provider.lower().replace(' ', '')}.com/{skill.lower().replace(' ', '-')}-course",
                "description": f"Learn {skill} from scratch with hands-on projects and real-world examples.",
                "prerequisites": random.sample(skills, random.randint(0, 2)) if random.random() > 0.5 else [],
                "certificate": random.random() > 0.3,
                "updated_at": self.fake.date_time_between(start_date="-2y", end_date="now")
            }
            resources.append(resource)
        
        return resources
    
    def create_test_resume_file(self, profile_data: Dict[str, Any]) -> str:
        """Create a temporary resume file for testing."""
        resume_text = self.generate_resume_text(profile_data)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(resume_text)
            return f.name
    
    def generate_performance_test_data(self, num_users: int = 100) -> Dict[str, Any]:
        """Generate data for performance testing."""
        return {
            "users": self.generate_user_data(num_users),
            "profiles": self.generate_profile_data(num_users),
            "jobs": self.generate_job_postings(num_users * 2),
            "learning_resources": self.generate_learning_resources(
                [skill for skills in self.skill_categories.values() for skill in skills],
                num_users
            )
        }
    
    def save_test_data_to_json(self, data: Dict[str, Any], filename: str):
        """Save test data to JSON file."""
        # Convert datetime objects to strings for JSON serialization
        def json_serializer(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=json_serializer)
    
    def generate_ml_test_datasets(self) -> Dict[str, Any]:
        """Generate datasets for ML model testing."""
        # Synthetic resume data for skill extraction testing
        resume_texts = []
        for _ in range(50):
            profile = self.generate_profile_data(1)[0]
            resume_text = self.generate_resume_text(profile)
            resume_texts.append({
                "text": resume_text,
                "expected_skills": profile["skills"],
                "experience_years": profile["experience_years"],
                "target_role": profile["dream_job"]
            })
        
        # Job-candidate matching data
        job_candidate_pairs = []
        jobs = self.generate_job_postings(20)
        profiles = self.generate_profile_data(30)
        
        for job in jobs:
            for profile in profiles:
                # Calculate similarity score based on skill overlap
                job_skills = set(job["required_skills"])
                candidate_skills = set(profile["skills"])
                overlap = len(job_skills & candidate_skills)
                total_skills = len(job_skills | candidate_skills)
                similarity = overlap / total_skills if total_skills > 0 else 0
                
                job_candidate_pairs.append({
                    "job": job,
                    "candidate": profile,
                    "similarity_score": similarity,
                    "is_good_match": similarity > 0.6
                })
        
        return {
            "resume_skill_extraction": resume_texts,
            "job_candidate_matching": job_candidate_pairs,
            "skill_embeddings": {
                skill: [random.random() for _ in range(384)]  # Mock 384-dim embeddings
                for skills in self.skill_categories.values()
                for skill in skills
            }
        }


# Utility functions for easy access
def create_test_users(count: int = 10) -> List[Dict[str, Any]]:
    """Quick function to create test users."""
    generator = TestDataGenerator()
    return generator.generate_user_data(count)


def create_test_profiles(count: int = 10) -> List[Dict[str, Any]]:
    """Quick function to create test profiles."""
    generator = TestDataGenerator()
    return generator.generate_profile_data(count)


def create_test_jobs(count: int = 20) -> List[Dict[str, Any]]:
    """Quick function to create test job postings."""
    generator = TestDataGenerator()
    return generator.generate_job_postings(count)


def create_performance_test_data(num_users: int = 100) -> Dict[str, Any]:
    """Quick function to create performance test data."""
    generator = TestDataGenerator()
    return generator.generate_performance_test_data(num_users)


if __name__ == "__main__":
    # Example usage
    generator = TestDataGenerator()
    
    # Generate sample data
    users = generator.generate_user_data(5)
    profiles = generator.generate_profile_data(5)
    jobs = generator.generate_job_postings(10)
    
    print("Generated test data:")
    print(f"- {len(users)} users")
    print(f"- {len(profiles)} profiles")
    print(f"- {len(jobs)} job postings")
    
    # Save to files
    generator.save_test_data_to_json({"users": users}, "test_users.json")
    generator.save_test_data_to_json({"profiles": profiles}, "test_profiles.json")
    generator.save_test_data_to_json({"jobs": jobs}, "test_jobs.json")
    
    print("Test data saved to JSON files!")