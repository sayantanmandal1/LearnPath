#!/usr/bin/env python3
"""
Advanced Career Path Analysis using Gemini API
This script demonstrates how to use the Gemini API for intelligent career analysis
"""
import asyncio
import sys
from pathlib import Path
import httpx
import json

# Add the backend directory to Python path
backend_dir = Path(__file__).parent
if str(backend_dir) not in sys.path:
    sys.path.append(str(backend_dir))

from app.core.config import settings


class GeminiCareerAnalyzer:
    """Advanced career analysis using Gemini API"""
    
    def __init__(self):
        self.api_key = settings.GEMINI_API_KEY
        self.api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"
        
    async def analyze_career_path(self, profile_data: dict) -> dict:
        """
        Analyze career path and provide recommendations using Gemini API
        
        Args:
            profile_data: Dictionary containing user profile information
            
        Returns:
            Dictionary with career analysis and recommendations
        """
        if not self.api_key:
            return self._fallback_analysis(profile_data)
            
        prompt = self._build_career_analysis_prompt(profile_data)
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.api_url}?key={self.api_key}",
                    json={
                        "contents": [{
                            "parts": [{
                                "text": prompt
                            }]
                        }]
                    },
                    headers={
                        "Content-Type": "application/json"
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if "candidates" in result and len(result["candidates"]) > 0:
                        analysis_text = result["candidates"][0]["content"]["parts"][0]["text"]
                        return self._parse_gemini_response(analysis_text)
                    else:
                        print(f"⚠️ Gemini API returned unexpected format: {result}")
                        return self._fallback_analysis(profile_data)
                else:
                    print(f"⚠️ Gemini API error: {response.status_code} - {response.text}")
                    return self._fallback_analysis(profile_data)
                    
        except Exception as e:
            print(f"⚠️ Error calling Gemini API: {str(e)}")
            return self._fallback_analysis(profile_data)
    
    def _build_career_analysis_prompt(self, profile_data: dict) -> str:
        """Build a comprehensive prompt for career analysis"""
        
        skills = profile_data.get('skills', [])
        experience = profile_data.get('experience_years', 0)
        current_role = profile_data.get('current_role', 'Unknown')
        industry = profile_data.get('industry', 'Technology')
        location = profile_data.get('location', 'Remote')
        career_goals = profile_data.get('career_goals', 'Career advancement')
        
        prompt = f"""
As an AI career advisor with expertise in technology careers, provide a comprehensive career path analysis for the following profile:

**Current Profile:**
- Current Role: {current_role}
- Years of Experience: {experience}
- Industry: {industry}
- Location: {location}
- Career Goals: {career_goals}
- Skills: {', '.join(skills)}

Please provide your analysis in the following JSON format:

```json
{{
    "career_assessment": {{
        "current_level": "Junior/Mid-level/Senior",
        "strengths": ["list", "of", "key", "strengths"],
        "growth_areas": ["areas", "for", "improvement"],
        "market_position": "assessment of market competitiveness"
    }},
    "recommended_paths": [
        {{
            "title": "Career Path Title",
            "timeline": "1-2 years / 2-3 years / 3-5 years",
            "probability": "High/Medium/Low",
            "steps": ["specific", "actionable", "steps"],
            "required_skills": ["skills", "to", "develop"],
            "salary_range": "$XX,000 - $XX,000",
            "reasoning": "Why this path makes sense"
        }}
    ],
    "skill_development": {{
        "immediate_priorities": ["skills", "to", "focus", "on", "now"],
        "medium_term_goals": ["skills", "for", "6-12", "months"],
        "long_term_investments": ["skills", "for", "career", "growth"],
        "learning_resources": ["specific", "course", "or", "certification", "recommendations"]
    }},
    "market_insights": {{
        "trending_skills": ["hot", "skills", "in", "the", "market"],
        "industry_outlook": "Brief outlook for the industry",
        "salary_trends": "Salary trend analysis",
        "location_factors": "How location affects opportunities"
    }},
    "action_plan": {{
        "next_30_days": ["immediate", "action", "items"],
        "next_90_days": ["short", "term", "goals"],
        "next_year": ["annual", "career", "objectives"]
    }}
}}
```

Please ensure the analysis is:
1. Specific to the user's current situation
2. Based on current market trends in technology
3. Realistic and actionable
4. Comprehensive but focused
5. Formatted exactly as the JSON structure above
        """
        
        return prompt
    
    def _parse_gemini_response(self, response_text: str) -> dict:
        """Parse Gemini's response and extract structured data"""
        try:
            # Try to extract JSON from the response
            start_json = response_text.find('{')
            end_json = response_text.rfind('}') + 1
            
            if start_json != -1 and end_json > start_json:
                json_str = response_text[start_json:end_json]
                return json.loads(json_str)
            else:
                # If no valid JSON found, create a structured response from the text
                return {
                    "career_assessment": {
                        "analysis": response_text[:500] + "..." if len(response_text) > 500 else response_text
                    },
                    "gemini_raw_response": response_text
                }
        except json.JSONDecodeError:
            return {
                "career_assessment": {
                    "analysis": response_text[:500] + "..." if len(response_text) > 500 else response_text
                },
                "parsing_error": "Could not parse JSON response",
                "gemini_raw_response": response_text
            }
    
    def _fallback_analysis(self, profile_data: dict) -> dict:
        """Provide basic analysis when Gemini API is not available"""
        skills = profile_data.get('skills', [])
        experience = profile_data.get('experience_years', 0)
        current_role = profile_data.get('current_role', 'Developer')
        
        # Basic career path recommendations based on skills
        paths = []
        
        if any(skill.lower() in ['python', 'fastapi', 'django', 'flask'] for skill in skills):
            paths.append({
                "title": "Senior Backend Developer",
                "timeline": "1-2 years",
                "probability": "High",
                "reasoning": "Strong Python backend skills provide clear advancement path"
            })
        
        if any(skill.lower() in ['aws', 'docker', 'kubernetes', 'devops'] for skill in skills):
            paths.append({
                "title": "DevOps Engineer / Cloud Architect",
                "timeline": "2-3 years",
                "probability": "Medium",
                "reasoning": "Cloud and containerization skills are highly valuable"
            })
        
        if any(skill.lower() in ['react', 'frontend', 'javascript', 'typescript'] for skill in skills):
            paths.append({
                "title": "Full Stack Developer",
                "timeline": "1-2 years",
                "probability": "High",
                "reasoning": "Frontend skills complement backend experience"
            })
        
        return {
            "career_assessment": {
                "current_level": "Mid-level" if experience > 2 else "Junior",
                "strengths": skills[:5],
                "analysis": f"Based on {experience} years of experience in {current_role}"
            },
            "recommended_paths": paths,
            "skill_development": {
                "immediate_priorities": ["System Design", "Advanced Programming Concepts"],
                "medium_term_goals": ["Leadership Skills", "Architecture Patterns"],
                "learning_resources": ["System Design Interview courses", "Cloud certifications"]
            },
            "fallback_analysis": True
        }


async def demo_career_analysis():
    """Demonstrate the Gemini-powered career analysis"""
    
    print("🎯 Advanced Career Path Analysis with Gemini AI")
    print("=" * 60)
    
    # Sample user profiles for testing
    profiles = [
        {
            "name": "Backend Developer Profile",
            "current_role": "Backend Developer",
            "experience_years": 3,
            "industry": "Technology",
            "location": "Remote",
            "career_goals": "Become a Senior Developer and eventually a Tech Lead",
            "skills": ["Python", "FastAPI", "PostgreSQL", "Redis", "AWS", "Docker", "Git", "REST APIs", "JWT Authentication", "Unit Testing"]
        },
        {
            "name": "Frontend Developer Profile", 
            "current_role": "Frontend Developer",
            "experience_years": 2,
            "industry": "E-commerce",
            "location": "New York",
            "career_goals": "Transition to Full Stack Development",
            "skills": ["React", "TypeScript", "JavaScript", "HTML", "CSS", "Node.js", "MongoDB", "Redux", "Webpack", "Jest"]
        }
    ]
    
    analyzer = GeminiCareerAnalyzer()
    
    for i, profile in enumerate(profiles, 1):
        print(f"\n🔍 Analysis {i}: {profile['name']}")
        print("-" * 50)
        
        # Perform career analysis
        analysis = await analyzer.analyze_career_path(profile)
        
        # Display results
        print(f"📊 Career Assessment:")
        career_assessment = analysis.get('career_assessment', {})
        if 'current_level' in career_assessment:
            print(f"  Current Level: {career_assessment['current_level']}")
        if 'strengths' in career_assessment:
            print(f"  Key Strengths: {', '.join(career_assessment['strengths'][:3])}")
        if 'analysis' in career_assessment:
            print(f"  Analysis: {career_assessment['analysis'][:150]}...")
        
        print(f"\n🚀 Recommended Career Paths:")
        paths = analysis.get('recommended_paths', [])
        for j, path in enumerate(paths[:3], 1):
            print(f"  {j}. {path.get('title', 'Unknown Path')}")
            print(f"     Timeline: {path.get('timeline', 'TBD')}")
            print(f"     Probability: {path.get('probability', 'Medium')}")
            if 'reasoning' in path:
                print(f"     Reasoning: {path['reasoning'][:100]}...")
        
        print(f"\n📚 Skill Development Recommendations:")
        skill_dev = analysis.get('skill_development', {})
        if 'immediate_priorities' in skill_dev:
            print(f"  Immediate Focus: {', '.join(skill_dev['immediate_priorities'][:3])}")
        if 'learning_resources' in skill_dev:
            print(f"  Learning Resources: {', '.join(skill_dev['learning_resources'][:2])}")
        
        if 'fallback_analysis' in analysis:
            print(f"\n⚠️ Using fallback analysis (Gemini API not available)")
        
        print(f"\n" + "="*50)


if __name__ == "__main__":
    print("🎯 Gemini AI Career Path Analysis Demo")
    print("=" * 60)
    print(f"API Key Configured: {'✅ Yes' if settings.GEMINI_API_KEY else '❌ No'}")
    print(f"Backend Server: http://127.0.0.1:8000")
    print("=" * 60)
    
    asyncio.run(demo_career_analysis())
    
    print("\n✅ Career Analysis Demo Complete!")
    print("\n💡 Next Steps:")
    print("  1. Visit http://127.0.0.1:8000/api/v1/docs to see all API endpoints")
    print("  2. Create a user account and profile to get personalized recommendations")
    print("  3. Upload your resume to get AI-powered career analysis")
    print("  4. Explore different career paths and skill development plans")