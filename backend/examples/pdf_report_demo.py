"""
PDF Report Generation Demo

This script demonstrates the comprehensive PDF report generation functionality
for career analysis reports with charts and visualizations.
"""
import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime

# Add the backend directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from app.services.pdf_report_service import PDFReportService
from app.schemas.analytics import (
    CareerAnalysisReport, PDFReportRequest,
    SkillRadarChart, CareerRoadmapVisualization, CareerRoadmapNode, CareerRoadmapEdge,
    SkillGapReport, SkillGapAnalysis, JobCompatibilityReport, JobCompatibilityScore,
    HistoricalProgressReport, ProgressTrackingEntry
)


def create_sample_career_report() -> CareerAnalysisReport:
    """Create a comprehensive sample career analysis report"""
    
    # Sample skill radar chart
    skill_radar = SkillRadarChart(
        user_id="demo-user-456",
        categories=[
            "Programming Languages", 
            "Web Frameworks", 
            "Databases", 
            "Cloud & DevOps", 
            "Soft Skills",
            "Machine Learning"
        ],
        user_scores=[88.0, 82.0, 75.0, 65.0, 85.0, 45.0],
        market_average=[78.0, 72.0, 70.0, 68.0, 75.0, 60.0],
        target_scores=[95.0, 90.0, 85.0, 80.0, 90.0, 75.0]
    )
    
    # Sample career roadmap with multiple milestones
    nodes = [
        CareerRoadmapNode(
            id="current",
            title="Senior Software Engineer",
            description="Current position with 5+ years experience",
            position={"x": 0, "y": 0},
            node_type="current",
            timeline_months=0,
            completion_status="completed"
        ),
        CareerRoadmapNode(
            id="milestone_1",
            title="Tech Lead",
            description="Lead a team of 3-5 developers",
            position={"x": 200, "y": 0},
            node_type="milestone",
            timeline_months=8,
            required_skills=["Team Leadership", "System Architecture", "Mentoring"],
            completion_status="in_progress"
        ),
        CareerRoadmapNode(
            id="milestone_2",
            title="Senior Tech Lead",
            description="Lead multiple teams and complex projects",
            position={"x": 400, "y": 0},
            node_type="milestone",
            timeline_months=18,
            required_skills=["Strategic Planning", "Cross-team Collaboration", "Technical Vision"],
            completion_status="not_started"
        ),
        CareerRoadmapNode(
            id="target",
            title="Engineering Manager",
            description="Manage engineering teams and drive technical strategy",
            position={"x": 600, "y": 0},
            node_type="target",
            timeline_months=30,
            completion_status="not_started"
        ),
        CareerRoadmapNode(
            id="alternative_1",
            title="Principal Engineer",
            description="Technical leadership without people management",
            position={"x": 400, "y": 150},
            node_type="alternative",
            timeline_months=24,
            required_skills=["Deep Technical Expertise", "Technical Mentoring", "Innovation"],
            completion_status="not_started"
        )
    ]
    
    edges = [
        CareerRoadmapEdge(
            id="edge_1",
            source_id="current",
            target_id="milestone_1",
            edge_type="direct",
            difficulty=0.6,
            estimated_duration_months=8,
            required_actions=["Complete leadership training", "Start mentoring junior developers"]
        ),
        CareerRoadmapEdge(
            id="edge_2",
            source_id="milestone_1",
            target_id="milestone_2",
            edge_type="direct",
            difficulty=0.7,
            estimated_duration_months=10,
            required_actions=["Lead cross-functional projects", "Develop strategic thinking skills"]
        ),
        CareerRoadmapEdge(
            id="edge_3",
            source_id="milestone_2",
            target_id="target",
            edge_type="direct",
            difficulty=0.8,
            estimated_duration_months=12,
            required_actions=["Complete management training", "Build stakeholder relationships"]
        ),
        CareerRoadmapEdge(
            id="alt_edge_1",
            source_id="milestone_1",
            target_id="alternative_1",
            edge_type="alternative",
            difficulty=0.7,
            estimated_duration_months=16,
            required_actions=["Deepen technical expertise", "Lead technical initiatives"]
        )
    ]
    
    career_roadmap = CareerRoadmapVisualization(
        user_id="demo-user-456",
        nodes=nodes,
        edges=edges,
        metadata={
            "target_role": "Engineering Manager",
            "total_timeline_months": 30,
            "difficulty_level": "intermediate-advanced",
            "alternative_paths": 1
        }
    )
    
    # Comprehensive skill gap analysis
    skill_gaps = [
        SkillGapAnalysis(
            skill_name="Team Leadership",
            current_level=45.0,
            target_level=85.0,
            gap_size=40.0,
            priority="high",
            estimated_learning_hours=160,
            recommended_resources=[
                {"type": "course", "name": "Engineering Leadership Fundamentals", "provider": "Coursera"},
                {"type": "book", "name": "The Manager's Path", "author": "Camille Fournier"},
                {"type": "workshop", "name": "Leadership Skills for Tech Professionals"}
            ],
            market_demand=0.95,
            salary_impact=25000.0
        ),
        SkillGapAnalysis(
            skill_name="System Architecture",
            current_level=60.0,
            target_level=90.0,
            gap_size=30.0,
            priority="high",
            estimated_learning_hours=120,
            recommended_resources=[
                {"type": "course", "name": "System Design Interview", "provider": "Educative"},
                {"type": "book", "name": "Designing Data-Intensive Applications", "author": "Martin Kleppmann"},
                {"type": "practice", "name": "Design large-scale systems"}
            ],
            market_demand=0.92,
            salary_impact=20000.0
        ),
        SkillGapAnalysis(
            skill_name="Machine Learning",
            current_level=45.0,
            target_level=75.0,
            gap_size=30.0,
            priority="medium",
            estimated_learning_hours=200,
            recommended_resources=[
                {"type": "course", "name": "Machine Learning Specialization", "provider": "Coursera"},
                {"type": "project", "name": "Build ML pipeline for production"},
                {"type": "certification", "name": "AWS Machine Learning Specialty"}
            ],
            market_demand=0.88,
            salary_impact=18000.0
        ),
        SkillGapAnalysis(
            skill_name="Strategic Planning",
            current_level=35.0,
            target_level=80.0,
            gap_size=45.0,
            priority="medium",
            estimated_learning_hours=100,
            recommended_resources=[
                {"type": "course", "name": "Strategic Thinking for Leaders", "provider": "LinkedIn Learning"},
                {"type": "mentorship", "name": "Find senior engineering manager mentor"},
                {"type": "practice", "name": "Lead quarterly planning sessions"}
            ],
            market_demand=0.85,
            salary_impact=15000.0
        ),
        SkillGapAnalysis(
            skill_name="Kubernetes",
            current_level=50.0,
            target_level=80.0,
            gap_size=30.0,
            priority="low",
            estimated_learning_hours=80,
            recommended_resources=[
                {"type": "course", "name": "Kubernetes for Developers", "provider": "Udemy"},
                {"type": "hands-on", "name": "Deploy applications to K8s cluster"},
                {"type": "certification", "name": "Certified Kubernetes Application Developer"}
            ],
            market_demand=0.82,
            salary_impact=12000.0
        )
    ]
    
    skill_gap_report = SkillGapReport(
        user_id="demo-user-456",
        target_role="Engineering Manager",
        overall_match_score=68.5,
        skill_gaps=skill_gaps,
        strengths=[
            "Python Programming", "FastAPI", "PostgreSQL", "Docker", 
            "Problem Solving", "Code Review", "Technical Documentation"
        ],
        total_learning_hours=660,
        priority_skills=["Team Leadership", "System Architecture"]
    )
    
    # Job compatibility analysis
    job_matches = [
        JobCompatibilityScore(
            job_id="job-001",
            job_title="Senior Software Engineer - Team Lead",
            company="TechCorp Inc.",
            overall_score=89.0,
            skill_match_score=85.0,
            experience_match_score=93.0,
            location_match_score=95.0,
            salary_match_score=88.0,
            matched_skills=["Python", "FastAPI", "PostgreSQL", "Docker", "Team Collaboration"],
            missing_skills=["Team Leadership", "Performance Management"],
            recommendation="apply"
        ),
        JobCompatibilityScore(
            job_id="job-002",
            job_title="Engineering Manager",
            company="Innovation Labs",
            overall_score=72.0,
            skill_match_score=65.0,
            experience_match_score=75.0,
            location_match_score=85.0,
            salary_match_score=82.0,
            matched_skills=["Python", "System Design", "Technical Leadership"],
            missing_skills=["Team Management", "Budget Planning", "Hiring Experience"],
            recommendation="consider"
        ),
        JobCompatibilityScore(
            job_id="job-003",
            job_title="Principal Software Engineer",
            company="DataTech Solutions",
            overall_score=81.0,
            skill_match_score=88.0,
            experience_match_score=78.0,
            location_match_score=75.0,
            salary_match_score=85.0,
            matched_skills=["Python", "System Architecture", "Machine Learning", "Mentoring"],
            missing_skills=["Deep Learning", "Research Experience"],
            recommendation="apply"
        ),
        JobCompatibilityScore(
            job_id="job-004",
            job_title="Tech Lead - Full Stack",
            company="StartupXYZ",
            overall_score=76.0,
            skill_match_score=80.0,
            experience_match_score=72.0,
            location_match_score=70.0,
            salary_match_score=78.0,
            matched_skills=["Python", "React", "PostgreSQL", "Leadership"],
            missing_skills=["Frontend Architecture", "Mobile Development"],
            recommendation="consider"
        ),
        JobCompatibilityScore(
            job_id="job-005",
            job_title="Senior Backend Engineer",
            company="Enterprise Corp",
            overall_score=85.0,
            skill_match_score=92.0,
            experience_match_score=88.0,
            location_match_score=80.0,
            salary_match_score=75.0,
            matched_skills=["Python", "FastAPI", "PostgreSQL", "Microservices", "Docker"],
            missing_skills=["Enterprise Integration", "Legacy System Migration"],
            recommendation="apply"
        )
    ]
    
    job_compatibility_report = JobCompatibilityReport(
        user_id="demo-user-456",
        job_matches=job_matches,
        filters_applied={
            "location": "San Francisco Bay Area",
            "salary_min": 150000,
            "experience_level": "senior",
            "remote_friendly": True
        },
        total_jobs_analyzed=127
    )
    
    # Progress tracking over the last 6 months
    skill_improvements = [
        ProgressTrackingEntry(
            user_id="demo-user-456",
            skill_name="Python",
            previous_score=82.0,
            current_score=88.0,
            improvement=6.0,
            tracking_period_days=180,
            evidence="Completed advanced Python course, contributed to open source projects",
            milestone_achieved="Python Expert Certification"
        ),
        ProgressTrackingEntry(
            user_id="demo-user-456",
            skill_name="System Design",
            previous_score=55.0,
            current_score=70.0,
            improvement=15.0,
            tracking_period_days=180,
            evidence="Designed microservices architecture for new product feature",
            milestone_achieved="Led system design review sessions"
        ),
        ProgressTrackingEntry(
            user_id="demo-user-456",
            skill_name="Docker",
            previous_score=70.0,
            current_score=78.0,
            improvement=8.0,
            tracking_period_days=180,
            evidence="Containerized legacy applications, optimized Docker builds"
        ),
        ProgressTrackingEntry(
            user_id="demo-user-456",
            skill_name="Team Collaboration",
            previous_score=78.0,
            current_score=85.0,
            improvement=7.0,
            tracking_period_days=180,
            evidence="Led cross-team integration project, improved team communication",
            milestone_achieved="Team Collaboration Excellence Award"
        ),
        ProgressTrackingEntry(
            user_id="demo-user-456",
            skill_name="Machine Learning",
            previous_score=35.0,
            current_score=45.0,
            improvement=10.0,
            tracking_period_days=180,
            evidence="Completed ML fundamentals course, built recommendation system prototype"
        )
    ]
    
    progress_report = HistoricalProgressReport(
        user_id="demo-user-456",
        tracking_period_days=180,
        skill_improvements=skill_improvements,
        overall_improvement_score=9.2,
        milestones_achieved=[
            "Python Expert Certification",
            "Led system design review sessions",
            "Team Collaboration Excellence Award",
            "Mentored 2 junior developers",
            "Completed leadership fundamentals course"
        ],
        trend_analysis={
            "trend": "strongly_improving",
            "velocity": "high",
            "focus_areas": ["technical_leadership", "system_design"],
            "improvement_rate": 9.2,
            "consistency_score": 0.85,
            "prediction": "On track to reach Tech Lead level within 8 months"
        }
    )
    
    # Comprehensive recommendations
    recommendations = [
        "Prioritize developing team leadership skills through formal training and hands-on experience mentoring junior developers",
        "Deepen system architecture knowledge by leading design reviews and studying large-scale system patterns",
        "Build strategic thinking capabilities by participating in product planning and technical roadmap discussions",
        "Gain experience with machine learning technologies to stay competitive in the evolving tech landscape",
        "Develop cross-functional collaboration skills by working closely with product and design teams",
        "Consider pursuing engineering management track given strong technical foundation and growing leadership interest",
        "Strengthen communication skills through technical writing, conference speaking, or internal tech talks",
        "Build a professional network in engineering leadership through industry meetups and conferences"
    ]
    
    # Actionable next steps
    next_steps = [
        "Enroll in 'Engineering Leadership Fundamentals' course within the next 2 weeks",
        "Schedule monthly 1:1s with current manager to discuss leadership development opportunities",
        "Volunteer to lead the next major system design initiative or architecture review",
        "Start mentoring at least one junior developer and document the experience",
        "Join or create an internal engineering leadership discussion group",
        "Complete a machine learning project that demonstrates practical application in your domain",
        "Apply for 2-3 Tech Lead positions to gain interview experience and market insights",
        "Set up quarterly skill assessments to track progress against career goals"
    ]
    
    # Create comprehensive report
    return CareerAnalysisReport(
        user_id="demo-user-456",
        profile_summary={
            "first_name": "Sarah",
            "last_name": "Johnson",
            "current_role": "Senior Software Engineer",
            "experience_level": "Senior (5+ years)",
            "location": "San Francisco, CA",
            "education": "MS Computer Science, Stanford University",
            "skills": [
                "Python", "FastAPI", "PostgreSQL", "React", "Docker", "Kubernetes",
                "System Design", "Microservices", "AWS", "Git", "Agile", "Code Review",
                "Technical Documentation", "Problem Solving", "Team Collaboration"
            ],
            "certifications": ["AWS Solutions Architect", "Python Expert Certification"],
            "years_experience": 6,
            "target_role": "Engineering Manager",
            "salary_range": {"min": 180000, "max": 220000},
            "remote_preference": "hybrid"
        },
        skill_radar_chart=skill_radar,
        career_roadmap=career_roadmap,
        skill_gap_report=skill_gap_report,
        job_compatibility_report=job_compatibility_report,
        progress_report=progress_report,
        recommendations=recommendations,
        next_steps=next_steps,
        report_version="2.0"
    )


async def demonstrate_pdf_generation():
    """Demonstrate PDF report generation with various configurations"""
    
    print("🚀 PDF Report Generation Demo")
    print("=" * 50)
    
    # Create sample data
    print("📊 Creating sample career analysis data...")
    career_report = create_sample_career_report()
    
    # Initialize PDF service
    print("🔧 Initializing PDF report service...")
    pdf_service = PDFReportService(storage_path="demo_reports")
    
    # Test different report configurations
    report_configs = [
        {
            "name": "Comprehensive Report with Charts",
            "request": PDFReportRequest(
                user_id="demo-user-456",
                report_type="comprehensive",
                include_charts=True,
                include_recommendations=True,
                branding={"company_name": "Career AI Pro", "logo_url": None}
            )
        },
        {
            "name": "Skills-Only Report",
            "request": PDFReportRequest(
                user_id="demo-user-456",
                report_type="skills_only",
                include_charts=True,
                include_recommendations=False
            )
        },
        {
            "name": "Career Roadmap Report",
            "request": PDFReportRequest(
                user_id="demo-user-456",
                report_type="career_only",
                include_charts=True,
                include_recommendations=True
            )
        },
        {
            "name": "Progress Tracking Report",
            "request": PDFReportRequest(
                user_id="demo-user-456",
                report_type="progress_only",
                include_charts=False,
                include_recommendations=True
            )
        }
    ]
    
    generated_reports = []
    
    for config in report_configs:
        print(f"\n📄 Generating {config['name']}...")
        
        try:
            response = await pdf_service.generate_comprehensive_report(
                career_report, config["request"]
            )
            
            generated_reports.append({
                "name": config["name"],
                "response": response,
                "file_path": await pdf_service.get_report_file(response.report_id)
            })
            
            print(f"   ✅ Success!")
            print(f"   📁 Report ID: {response.report_id}")
            print(f"   📊 File Size: {response.file_size_bytes:,} bytes")
            print(f"   📖 Pages: {response.page_count}")
            print(f"   🔗 Download URL: {response.file_url}")
            print(f"   ⏰ Expires: {response.expires_at}")
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
    
    # Display summary
    print(f"\n📋 Generation Summary")
    print("=" * 30)
    print(f"Total reports generated: {len(generated_reports)}")
    
    total_size = sum(report["response"].file_size_bytes for report in generated_reports)
    total_pages = sum(report["response"].page_count for report in generated_reports)
    
    print(f"Total file size: {total_size:,} bytes ({total_size/1024/1024:.2f} MB)")
    print(f"Total pages: {total_pages}")
    
    # List generated files
    print(f"\n📁 Generated Files:")
    for report in generated_reports:
        if report["file_path"] and report["file_path"].exists():
            print(f"   • {report['name']}: {report['file_path'].name}")
        else:
            print(f"   • {report['name']}: File not found")
    
    # Demonstrate cleanup
    print(f"\n🧹 Testing cleanup functionality...")
    try:
        await pdf_service.cleanup_expired_reports()
        print("   ✅ Cleanup completed successfully")
    except Exception as e:
        print(f"   ❌ Cleanup error: {str(e)}")
    
    print(f"\n🎉 Demo completed successfully!")
    print(f"Check the 'demo_reports' directory for generated PDF files.")


async def demonstrate_error_handling():
    """Demonstrate error handling scenarios"""
    
    print("\n🚨 Error Handling Demo")
    print("=" * 30)
    
    pdf_service = PDFReportService(storage_path="demo_reports")
    
    # Test with invalid report ID
    print("Testing invalid report ID retrieval...")
    file_path = await pdf_service.get_report_file("invalid-report-id")
    if file_path is None:
        print("   ✅ Correctly handled invalid report ID")
    else:
        print("   ❌ Unexpected result for invalid report ID")
    
    # Test with minimal data - create minimal objects instead of None
    print("Testing with minimal career data...")
    
    # Create minimal skill radar chart
    minimal_skill_radar = SkillRadarChart(
        user_id="minimal-user",
        categories=["Programming"],
        user_scores=[50.0],
        market_average=[60.0]
    )
    
    # Create minimal career roadmap
    minimal_nodes = [
        CareerRoadmapNode(
            id="current",
            title="Current Role",
            description="Current position",
            position={"x": 0, "y": 0},
            node_type="current",
            timeline_months=0,
            completion_status="completed"
        )
    ]
    
    minimal_roadmap = CareerRoadmapVisualization(
        user_id="minimal-user",
        nodes=minimal_nodes,
        edges=[],
        metadata={}
    )
    
    # Create minimal skill gap report
    minimal_skill_gap = SkillGapReport(
        user_id="minimal-user",
        target_role="Test Role",
        overall_match_score=50.0,
        skill_gaps=[],
        strengths=["Basic Skills"],
        total_learning_hours=0,
        priority_skills=[]
    )
    
    # Create minimal job compatibility report
    minimal_job_report = JobCompatibilityReport(
        user_id="minimal-user",
        job_matches=[],
        total_jobs_analyzed=0
    )
    
    # Create minimal progress report
    minimal_progress = HistoricalProgressReport(
        user_id="minimal-user",
        tracking_period_days=30,
        skill_improvements=[],
        overall_improvement_score=0.0,
        milestones_achieved=[],
        trend_analysis={}
    )
    
    minimal_report = CareerAnalysisReport(
        user_id="minimal-user",
        profile_summary={"first_name": "Test", "last_name": "User"},
        skill_radar_chart=minimal_skill_radar,
        career_roadmap=minimal_roadmap,
        skill_gap_report=minimal_skill_gap,
        job_compatibility_report=minimal_job_report,
        progress_report=minimal_progress,
        recommendations=[],
        next_steps=[]
    )
    
    minimal_request = PDFReportRequest(
        user_id="minimal-user",
        report_type="comprehensive",
        include_charts=True,
        include_recommendations=True
    )
    
    try:
        response = await pdf_service.generate_comprehensive_report(minimal_report, minimal_request)
        print("   ✅ Successfully handled minimal data")
        print(f"   📁 Generated report: {response.report_id}")
    except Exception as e:
        print(f"   ⚠️  Error with minimal data: {str(e)}")


if __name__ == "__main__":
    print("Starting PDF Report Generation Demo...")
    
    # Run the main demonstration
    asyncio.run(demonstrate_pdf_generation())
    
    # Run error handling demonstration
    asyncio.run(demonstrate_error_handling())
    
    print("\nDemo completed! 🎉")