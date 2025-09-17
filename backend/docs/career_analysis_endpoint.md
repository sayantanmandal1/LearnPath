# Career Analysis Endpoint

## Overview

The career analysis endpoint (`/api/v1/career-analysis/analyze`) processes career analysis form data from the frontend and generates comprehensive AI-powered recommendations.

## Endpoint Details

- **URL**: `/api/v1/career-analysis/analyze`
- **Method**: `POST`
- **Authentication**: Required (JWT Bearer token)
- **Content-Type**: `application/json`

## Request Format

The endpoint expects a JSON payload matching the frontend analyze form structure:

```json
{
  "current_role": "Software Developer",
  "experience": "2-3",
  "industry": "Technology",
  "location": "San Francisco, CA",
  "desired_role": "Senior Software Engineer",
  "career_goals": "I want to advance to a senior role and lead technical projects",
  "timeframe": "medium",
  "salary_expectation": "$90,000 - $120,000",
  "skills": "Python, JavaScript, React, SQL",
  "education": "Bachelor's in Computer Science",
  "certifications": "AWS Certified Developer",
  "languages": "English, Spanish",
  "work_type": "hybrid",
  "company_size": "medium",
  "work_culture": "Collaborative and innovative environment",
  "benefits": ["Health Insurance", "401(k) Matching", "Remote Work"]
}
```

### Required Fields

- `current_role`: Current job title
- `experience`: Years of experience (options: "0-1", "2-3", "4-6", "7-10", "10+")
- `industry`: Current industry
- `location`: Current location
- `desired_role`: Target job role
- `career_goals`: Career aspirations description
- `timeframe`: Career transition timeframe (options: "immediate", "short", "medium", "long")
- `salary_expectation`: Expected salary range
- `skills`: Comma-separated list of technical skills
- `education`: Educational background
- `work_type`: Preferred work arrangement (options: "remote", "hybrid", "onsite", "flexible")
- `company_size`: Preferred company size (options: "startup", "small", "medium", "large")

### Optional Fields

- `certifications`: Professional certifications (comma-separated)
- `languages`: Programming/spoken languages (comma-separated)
- `work_culture`: Work culture preferences
- `benefits`: Array of important benefits

## Response Format

The endpoint returns a comprehensive analysis matching the frontend expectations:

```json
{
  "overall_score": 85,
  "strengths": [
    "Strong technical skills in Python, JavaScript, React",
    "Solid professional foundation",
    "Strong educational background"
  ],
  "improvements": [
    "Consider learning cloud technologies (AWS, Azure) for Senior Software Engineer",
    "Develop project management skills",
    "Build a stronger professional network",
    "Gain experience in emerging technologies"
  ],
  "recommendations": [
    {
      "type": "job",
      "title": "Senior Software Developer",
      "company": "TechCorp Inc.",
      "match": 92,
      "salary": "$95,000 - $120,000",
      "location": "San Francisco, CA"
    }
  ],
  "learning_paths": [
    {
      "title": "Cloud Computing Fundamentals",
      "provider": "AWS",
      "duration": "6 weeks",
      "difficulty": "Intermediate"
    }
  ],
  "market_insights": {
    "demand_trend": "High",
    "salary_growth": "+15% YoY",
    "top_skills": ["Python", "JavaScript", "React", "AWS", "Docker"],
    "competition_level": "Medium"
  }
}
```

## Functionality

The endpoint performs the following operations:

1. **Data Processing**: Parses and normalizes the form data
2. **Profile Management**: Creates or updates the user's profile with the analysis data
3. **Analysis Generation**: Generates career recommendations, learning paths, and market insights
4. **Response Formatting**: Returns results in the format expected by the frontend

## Profile Data Storage

The endpoint automatically creates or updates the user's profile with the provided data:

- Parses experience years from string format to integer
- Converts comma-separated skills to structured skill data
- Stores work preferences and career goals
- Maintains profile history for tracking changes

## Error Handling

The endpoint handles various error scenarios:

- **401/403**: Authentication required
- **422**: Validation errors for missing or invalid fields
- **500**: Internal server errors during processing

## Integration with Frontend

This endpoint is designed to work seamlessly with the frontend analyze page:

1. Frontend collects user data through the multi-step form
2. Data is sent to this endpoint when user clicks "Analyze"
3. Backend processes the data and generates recommendations
4. Frontend displays the results in the analysis results page

## Future Enhancements

The current implementation uses mock data for recommendations. Future versions will integrate with:

- ML recommendation engine for personalized job matching
- Real-time job market data for accurate insights
- Learning path optimization algorithms
- Advanced skill gap analysis