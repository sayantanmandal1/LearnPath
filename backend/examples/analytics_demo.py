"""
Analytics service demonstration script
"""
import asyncio
import sys
import os
from pathlib import Path

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from app.services.analytics_service import AnalyticsService
from app.services.visualization_service import VisualizationService
# from app.services.pdf_report_service import PDFReportService
from app.schemas.analytics import (
    AnalyticsRequest, ChartConfiguration, ChartType, PDFReportRequest
)
# from app.core.database import get_async_session
from app.models.user import User
from app.models.profile import UserProfile
from app.models.skill import Skill, UserSkill


async def create_demo_data(db_session):
    """Create demo data for analytics testing"""
    print("Creating demo data...")
    
    # This would typically be done through the proper service layers
    # For demo purposes, we'll simulate the data
    print("Demo data created (simulated)")


async def demo_skill_radar_chart(analytics_service):
    """Demonstrate skill radar chart generation"""
    print("\n=== Skill Radar Chart Demo ===")
    
    try:
        radar_chart = await analytics_service.generate_skill_radar_chart(
            user_id="demo-user-123",
            target_role="Senior Developer"
        )
        
        print(f"Generated radar chart for user: {radar_chart.user_id}")
        print(f"Categories: {radar_chart.categories}")
        print(f"User scores: {radar_chart.user_scores}")
        print(f"Market averages: {radar_chart.market_average}")
        if radar_chart.target_scores:
            print(f"Target scores: {radar_chart.target_scores}")
        
        return radar_chart
        
    except Exception as e:
        print(f"Error generating radar chart: {e}")
        return None


async def demo_career_roadmap(analytics_service):
    """Demonstrate career roadmap generation"""
    print("\n=== Career Roadmap Demo ===")
    
    try:
        roadmap = await analytics_service.generate_career_roadmap(
            user_id="demo-user-123",
            target_role="Tech Lead"
        )
        
        print(f"Generated roadmap for user: {roadmap.user_id}")
        print(f"Number of nodes: {len(roadmap.nodes)}")
        print(f"Number of edges: {len(roadmap.edges)}")
        
        print("\nRoadmap nodes:")
        for node in roadmap.nodes:
            print(f"  - {node.title} ({node.node_type})")
            if node.timeline_months:
                print(f"    Timeline: {node.timeline_months} months")
        
        print("\nRoadmap edges:")
        for edge in roadmap.edges:
            print(f"  - {edge.source_id} -> {edge.target_id} ({edge.edge_type})")
            print(f"    Difficulty: {edge.difficulty:.1f}")
        
        return roadmap
        
    except Exception as e:
        print(f"Error generating roadmap: {e}")
        return None


async def demo_skill_gap_analysis(analytics_service):
    """Demonstrate skill gap analysis"""
    print("\n=== Skill Gap Analysis Demo ===")
    
    try:
        gap_report = await analytics_service.analyze_skill_gaps(
            user_id="demo-user-123",
            target_role="Senior Developer"
        )
        
        print(f"Skill gap analysis for: {gap_report.target_role}")
        print(f"Overall match score: {gap_report.overall_match_score:.1f}%")
        print(f"Total learning hours: {gap_report.total_learning_hours}")
        
        print(f"\nStrengths ({len(gap_report.strengths)}):")
        for strength in gap_report.strengths:
            print(f"  - {strength}")
        
        print(f"\nSkill gaps ({len(gap_report.skill_gaps)}):")
        for gap in gap_report.skill_gaps[:5]:  # Show top 5
            print(f"  - {gap.skill_name}: {gap.current_level:.1f}% -> {gap.target_level:.1f}% "
                  f"(Gap: {gap.gap_size:.1f}, Priority: {gap.priority})")
            if gap.estimated_learning_hours:
                print(f"    Estimated learning: {gap.estimated_learning_hours} hours")
        
        print(f"\nPriority skills: {', '.join(gap_report.priority_skills)}")
        
        return gap_report
        
    except Exception as e:
        print(f"Error analyzing skill gaps: {e}")
        return None


async def demo_job_compatibility(analytics_service):
    """Demonstrate job compatibility scoring"""
    print("\n=== Job Compatibility Demo ===")
    
    try:
        compatibility_report = await analytics_service.generate_job_compatibility_scores(
            user_id="demo-user-123",
            job_filters={"location": "San Francisco", "experience_level": "mid"},
            limit=10
        )
        
        print(f"Analyzed {compatibility_report.total_jobs_analyzed} jobs")
        print(f"Found {len(compatibility_report.job_matches)} matches")
        
        print("\nTop job matches:")
        for i, job in enumerate(compatibility_report.job_matches[:5], 1):
            print(f"  {i}. {job.job_title} at {job.company}")
            print(f"     Overall score: {job.overall_score:.1f}%")
            print(f"     Skill match: {job.skill_match_score:.1f}%")
            print(f"     Experience match: {job.experience_match_score:.1f}%")
            print(f"     Recommendation: {job.recommendation}")
            print(f"     Matched skills: {', '.join(job.matched_skills[:3])}")
            if job.missing_skills:
                print(f"     Missing skills: {', '.join(job.missing_skills[:3])}")
            print()
        
        return compatibility_report
        
    except Exception as e:
        print(f"Error generating job compatibility: {e}")
        return None


async def demo_progress_tracking(analytics_service):
    """Demonstrate progress tracking"""
    print("\n=== Progress Tracking Demo ===")
    
    try:
        progress_report = await analytics_service.track_historical_progress(
            user_id="demo-user-123",
            tracking_period_days=90
        )
        
        print(f"Progress tracking for {progress_report.tracking_period_days} days")
        print(f"Overall improvement score: {progress_report.overall_improvement_score:.1f}%")
        
        print(f"\nSkill improvements ({len(progress_report.skill_improvements)}):")
        for improvement in progress_report.skill_improvements:
            print(f"  - {improvement.skill_name}: "
                  f"{improvement.previous_score:.1f}% -> {improvement.current_score:.1f}% "
                  f"(+{improvement.improvement:.1f}%)")
        
        print(f"\nMilestones achieved ({len(progress_report.milestones_achieved)}):")
        for milestone in progress_report.milestones_achieved:
            print(f"  - {milestone}")
        
        print(f"\nTrend analysis:")
        trend = progress_report.trend_analysis
        print(f"  - Trend: {trend.get('trend', 'N/A')}")
        print(f"  - Velocity: {trend.get('velocity', 0):.1f}%")
        print(f"  - Skills improved: {trend.get('total_skills_improved', 0)}")
        if trend.get('best_performing_skill'):
            print(f"  - Best performing skill: {trend['best_performing_skill']}")
        
        return progress_report
        
    except Exception as e:
        print(f"Error tracking progress: {e}")
        return None


async def demo_comprehensive_report(analytics_service):
    """Demonstrate comprehensive analytics report"""
    print("\n=== Comprehensive Report Demo ===")
    
    try:
        request = AnalyticsRequest(
            user_id="demo-user-123",
            analysis_types=["full_report"],
            target_role="Senior Developer",
            include_job_matches=True,
            include_progress_tracking=True,
            tracking_period_days=90
        )
        
        report = await analytics_service.generate_comprehensive_report(request)
        
        print(f"Generated comprehensive report for user: {report.user_id}")
        print(f"Profile summary: {report.profile_summary.get('name', 'N/A')}")
        
        if report.skill_radar_chart:
            print("✓ Skill radar chart included")
        
        if report.career_roadmap:
            print("✓ Career roadmap included")
        
        if report.skill_gap_report:
            print(f"✓ Skill gap report included (Match: {report.skill_gap_report.overall_match_score:.1f}%)")
        
        if report.job_compatibility_report:
            print(f"✓ Job compatibility report included ({len(report.job_compatibility_report.job_matches)} matches)")
        
        if report.progress_report:
            print(f"✓ Progress report included (Improvement: {report.progress_report.overall_improvement_score:.1f}%)")
        
        print(f"\nRecommendations ({len(report.recommendations)}):")
        for i, rec in enumerate(report.recommendations, 1):
            print(f"  {i}. {rec}")
        
        print(f"\nNext steps ({len(report.next_steps)}):")
        for i, step in enumerate(report.next_steps, 1):
            print(f"  {i}. {step}")
        
        return report
        
    except Exception as e:
        print(f"Error generating comprehensive report: {e}")
        return None


async def demo_visualizations(viz_service, radar_chart, roadmap, gap_report):
    """Demonstrate visualization generation"""
    print("\n=== Visualizations Demo ===")
    
    try:
        config = ChartConfiguration(
            chart_type=ChartType.RADAR,
            title="Skills Analysis",
            width=800,
            height=600,
            color_scheme="professional"
        )
        
        if radar_chart:
            print("Generating radar chart visualization...")
            radar_viz = viz_service.generate_skill_radar_chart_data(radar_chart, config)
            print(f"✓ Radar chart data generated (type: {radar_viz['type']})")
        
        if roadmap:
            print("Generating roadmap visualization...")
            roadmap_viz = viz_service.generate_career_roadmap_data(roadmap, config)
            print(f"✓ Roadmap data generated ({len(roadmap_viz['nodes'])} nodes, {len(roadmap_viz['edges'])} edges)")
        
        if gap_report:
            print("Generating skill gap visualization...")
            gap_viz = viz_service.generate_skill_gap_chart_data(gap_report, config)
            print(f"✓ Skill gap chart data generated (type: {gap_viz['type']})")
        
    except Exception as e:
        print(f"Error generating visualizations: {e}")


async def demo_pdf_report(pdf_service, comprehensive_report):
    """Demonstrate PDF report generation"""
    print("\n=== PDF Report Demo ===")
    
    try:
        if not comprehensive_report:
            print("No comprehensive report available for PDF generation")
            return
        
        request = PDFReportRequest(
            user_id="demo-user-123",
            report_type="comprehensive",
            include_charts=True,
            include_recommendations=True
        )
        
        print("Generating PDF report...")
        pdf_response = await pdf_service.generate_comprehensive_report(
            comprehensive_report, request
        )
        
        print(f"✓ PDF report generated:")
        print(f"  - Report ID: {pdf_response.report_id}")
        print(f"  - File URL: {pdf_response.file_url}")
        print(f"  - File size: {pdf_response.file_size_bytes} bytes")
        print(f"  - Page count: {pdf_response.page_count}")
        print(f"  - Expires at: {pdf_response.expires_at}")
        
        # Test file retrieval
        report_file = await pdf_service.get_report_file(pdf_response.report_id)
        if report_file and report_file.exists():
            print(f"✓ PDF file exists at: {report_file}")
        else:
            print("✗ PDF file not found")
        
    except Exception as e:
        print(f"Error generating PDF report: {e}")


async def main():
    """Main demo function"""
    print("=== AI Career Recommender Analytics Demo ===")
    
    try:
        # Initialize services
        print("Initializing services...")
        
        # Use a mock database session for demo
        from unittest.mock import Mock
        mock_db = Mock()
        
        analytics_service = AnalyticsService(mock_db)
        viz_service = VisualizationService()
        # pdf_service = PDFReportService(storage_path="demo_reports")
        
        print("✓ Services initialized")
        
        # Create demo data (simulated)
        # await create_demo_data(mock_db)
        
        # Run analytics demos
        radar_chart = await demo_skill_radar_chart(analytics_service)
        roadmap = await demo_career_roadmap(analytics_service)
        gap_report = await demo_skill_gap_analysis(analytics_service)
        compatibility_report = await demo_job_compatibility(analytics_service)
        progress_report = await demo_progress_tracking(analytics_service)
        comprehensive_report = await demo_comprehensive_report(analytics_service)
        
        # Run visualization demos
        await demo_visualizations(viz_service, radar_chart, roadmap, gap_report)
        
        # Run PDF report demo (disabled)
        # await demo_pdf_report(pdf_service, comprehensive_report)
        
        print("\n=== Demo completed successfully! ===")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())