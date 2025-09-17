"""
PDF report generation service for comprehensive career analysis reports
"""
import uuid
import json
import base64
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging
import io

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.platypus.flowables import Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
    from reportlab.graphics.shapes import Drawing, Rect, String
    from reportlab.graphics.charts.piecharts import Pie
    from reportlab.graphics.charts.barcharts import VerticalBarChart
    from reportlab.graphics.charts.linecharts import HorizontalLineChart
    from reportlab.graphics import renderPDF
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    # Define dummy classes and functions for type hints when ReportLab is not available
    class Image:
        pass
    
    def getSampleStyleSheet():
        return None
    
    class ParagraphStyle:
        def __init__(self, *args, **kwargs):
            pass

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from app.schemas.analytics import (
    CareerAnalysisReport, PDFReportRequest, PDFReportResponse
)
from app.core.exceptions import PDFGenerationError

logger = logging.getLogger(__name__)


class PDFReportService:
    """Service for generating comprehensive PDF career analysis reports"""
    
    def __init__(self, storage_path: str = "reports"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        if not REPORTLAB_AVAILABLE:
            logger.warning("ReportLab not available. PDF generation will be disabled.")
        
        # Initialize styles
        self.styles = getSampleStyleSheet() if REPORTLAB_AVAILABLE else None
        if self.styles:
            self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles"""
        # Helper function to add style if it doesn't exist
        def add_style_if_not_exists(style):
            if style.name not in self.styles:
                self.styles.add(style)
        
        # Title style
        add_style_if_not_exists(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#2c3e50')
        ))
        
        # Section header style
        add_style_if_not_exists(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceBefore=20,
            spaceAfter=12,
            textColor=colors.HexColor('#34495e'),
            borderWidth=1,
            borderColor=colors.HexColor('#3498db'),
            borderPadding=5
        ))
        
        # Subsection style
        add_style_if_not_exists(ParagraphStyle(
            name='SubSection',
            parent=self.styles['Heading3'],
            fontSize=14,
            spaceBefore=15,
            spaceAfter=8,
            textColor=colors.HexColor('#2980b9')
        ))
        
        # Body text with better spacing
        add_style_if_not_exists(ParagraphStyle(
            name='CustomBodyText',  # Changed name to avoid conflict
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=8,
            alignment=TA_JUSTIFY
        ))
        
        # Highlight style for important information
        add_style_if_not_exists(ParagraphStyle(
            name='Highlight',
            parent=self.styles['Normal'],
            fontSize=12,
            textColor=colors.HexColor('#e74c3c'),
            backColor=colors.HexColor('#fdf2f2'),
            borderWidth=1,
            borderColor=colors.HexColor('#e74c3c'),
            borderPadding=8,
            spaceAfter=10
        ))
    
    async def generate_comprehensive_report(
        self, 
        report_data: CareerAnalysisReport, 
        request: PDFReportRequest
    ) -> PDFReportResponse:
        """Generate comprehensive PDF career analysis report"""
        if not REPORTLAB_AVAILABLE:
            raise PDFGenerationError("ReportLab library not available for PDF generation")
        
        try:
            report_id = str(uuid.uuid4())
            filename = f"career_report_{report_data.user_id}_{report_id}.pdf"
            file_path = self.storage_path / filename
            
            # Create PDF document
            doc = SimpleDocTemplate(
                str(file_path),
                pagesize=letter,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=18
            )
            
            # Build story (content)
            story = []
            
            # Add title page
            story.extend(self._create_title_page(report_data, request))
            story.append(PageBreak())
            
            # Add executive summary
            story.extend(self._create_executive_summary(report_data))
            story.append(PageBreak())
            
            # Add profile summary
            story.extend(self._create_profile_summary(report_data))
            
            # Add skill analysis if available
            if report_data.skill_radar_chart and request.include_charts:
                story.append(PageBreak())
                story.extend(self._create_skill_analysis_section(report_data))
            
            # Add career roadmap if available
            if report_data.career_roadmap and request.include_charts:
                story.append(PageBreak())
                story.extend(self._create_career_roadmap_section(report_data))
            
            # Add skill gap analysis if available
            if report_data.skill_gap_report:
                story.append(PageBreak())
                story.extend(self._create_skill_gap_section(report_data))
            
            # Add job compatibility if available
            if report_data.job_compatibility_report:
                story.append(PageBreak())
                story.extend(self._create_job_compatibility_section(report_data))
            
            # Add progress tracking if available
            if report_data.progress_report:
                story.append(PageBreak())
                story.extend(self._create_progress_section(report_data))
            
            # Add recommendations
            if report_data.recommendations and request.include_recommendations:
                story.append(PageBreak())
                story.extend(self._create_recommendations_section(report_data))
            
            # Add next steps
            if report_data.next_steps:
                story.extend(self._create_next_steps_section(report_data))
            
            # Build PDF
            doc.build(story)
            
            # Get file info
            file_size = file_path.stat().st_size
            expires_at = datetime.utcnow() + timedelta(days=7)
            
            # Count pages (approximate)
            page_count = len([item for item in story if isinstance(item, PageBreak)]) + 1
            
            return PDFReportResponse(
                report_id=report_id,
                file_url=f"/api/v1/analytics/reports/{report_id}/download",
                file_size_bytes=file_size,
                page_count=page_count,
                expires_at=expires_at
            )
            
        except Exception as e:
            logger.error(f"Error generating PDF report: {str(e)}")
            raise PDFGenerationError(f"Failed to generate PDF report: {str(e)}")
    
    def _create_title_page(self, report_data: CareerAnalysisReport, request: PDFReportRequest) -> List:
        """Create title page content"""
        content = []
        
        # Main title
        content.append(Spacer(1, 2*inch))
        content.append(Paragraph("Career Analysis Report", self.styles['CustomTitle']))
        content.append(Spacer(1, 0.5*inch))
        
        # User information
        profile = report_data.profile_summary
        user_name = f"{profile.get('first_name', '')} {profile.get('last_name', '')}".strip()
        if user_name:
            content.append(Paragraph(f"Prepared for: {user_name}", self.styles['Heading2']))
        
        current_role = profile.get('current_role', 'Professional')
        content.append(Paragraph(f"Current Role: {current_role}", self.styles['Normal']))
        content.append(Spacer(1, 0.3*inch))
        
        # Report metadata
        content.append(Paragraph(f"Generated: {datetime.utcnow().strftime('%B %d, %Y')}", self.styles['Normal']))
        content.append(Paragraph(f"Report Type: {request.report_type.title()}", self.styles['Normal']))
        content.append(Spacer(1, 1*inch))
        
        # Report summary stats
        if hasattr(report_data, 'skill_gap_report') and report_data.skill_gap_report:
            overall_score = report_data.skill_gap_report.overall_match_score
            content.append(Paragraph(f"Overall Career Match Score: {overall_score:.1f}%", self.styles['Highlight']))
        
        return content
    
    def _create_executive_summary(self, report_data: CareerAnalysisReport) -> List:
        """Create executive summary section"""
        content = []
        content.append(Paragraph("Executive Summary", self.styles['SectionHeader']))
        
        # Key insights
        insights = []
        
        if report_data.skill_gap_report:
            gap_report = report_data.skill_gap_report
            insights.append(f"• Overall career match score: {gap_report.overall_match_score:.1f}%")
            insights.append(f"• {len(gap_report.strengths)} key strengths identified")
            insights.append(f"• {len(gap_report.skill_gaps)} skill gaps to address")
            insights.append(f"• Estimated learning time: {gap_report.total_learning_hours} hours")
        
        if report_data.job_compatibility_report:
            job_report = report_data.job_compatibility_report
            top_match = max(job_report.job_matches, key=lambda x: x.overall_score) if job_report.job_matches else None
            if top_match:
                insights.append(f"• Best job match: {top_match.job_title} ({top_match.overall_score:.1f}% compatibility)")
        
        if report_data.progress_report:
            progress = report_data.progress_report
            insights.append(f"• Overall improvement score: {progress.overall_improvement_score:.1f}%")
            insights.append(f"• {len(progress.milestones_achieved)} milestones achieved")
        
        for insight in insights:
            content.append(Paragraph(insight, self.styles['CustomBodyText']))
        
        content.append(Spacer(1, 0.2*inch))
        
        # Key recommendations preview
        if report_data.recommendations:
            content.append(Paragraph("Key Recommendations:", self.styles['SubSection']))
            for i, rec in enumerate(report_data.recommendations[:3]):  # Top 3 recommendations
                content.append(Paragraph(f"{i+1}. {rec}", self.styles['CustomBodyText']))
        
        return content
    
    def _create_profile_summary(self, report_data: CareerAnalysisReport) -> List:
        """Create profile summary section"""
        content = []
        content.append(Paragraph("Profile Summary", self.styles['SectionHeader']))
        
        profile = report_data.profile_summary
        
        # Create profile table
        profile_data = [
            ['Field', 'Information'],
            ['Current Role', profile.get('current_role', 'Not specified')],
            ['Experience Level', profile.get('experience_level', 'Not specified')],
            ['Location', profile.get('location', 'Not specified')],
            ['Education', profile.get('education', 'Not specified')],
        ]
        
        if profile.get('skills'):
            skills_text = ', '.join(profile['skills'][:10])  # First 10 skills
            if len(profile['skills']) > 10:
                skills_text += f" (+{len(profile['skills']) - 10} more)"
            profile_data.append(['Key Skills', skills_text])
        
        profile_table = Table(profile_data, colWidths=[2*inch, 4*inch])
        profile_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
        ]))
        
        content.append(profile_table)
        return content
    
    def _create_skill_analysis_section(self, report_data: CareerAnalysisReport) -> List:
        """Create skill analysis section with charts"""
        content = []
        content.append(Paragraph("Skill Analysis", self.styles['SectionHeader']))
        
        radar_chart = report_data.skill_radar_chart
        
        # Create skill summary table
        skill_data = [['Skill Category', 'Your Score', 'Market Average', 'Gap']]
        
        for i, category in enumerate(radar_chart.categories):
            user_score = radar_chart.user_scores[i]
            market_avg = radar_chart.market_average[i]
            gap = user_score - market_avg
            gap_text = f"+{gap:.1f}" if gap > 0 else f"{gap:.1f}"
            
            skill_data.append([
                category,
                f"{user_score:.1f}%",
                f"{market_avg:.1f}%",
                gap_text
            ])
        
        skill_table = Table(skill_data, colWidths=[2.5*inch, 1*inch, 1*inch, 1*inch])
        skill_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2ecc71')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
        ]))
        
        content.append(skill_table)
        content.append(Spacer(1, 0.2*inch))
        
        # Add chart placeholder or generate simple visualization
        if MATPLOTLIB_AVAILABLE:
            chart_image = self._create_skill_radar_chart_image(radar_chart)
            if chart_image:
                content.append(chart_image)
        else:
            content.append(Paragraph("Skill Radar Chart: [Chart visualization requires matplotlib]", self.styles['Normal']))
        
        return content
    
    def _create_career_roadmap_section(self, report_data: CareerAnalysisReport) -> List:
        """Create career roadmap section"""
        content = []
        content.append(Paragraph("Career Roadmap", self.styles['SectionHeader']))
        
        roadmap = report_data.career_roadmap
        
        # Create roadmap summary
        content.append(Paragraph(f"Target Role: {roadmap.metadata.get('target_role', 'Not specified')}", self.styles['SubSection']))
        
        timeline = roadmap.metadata.get('total_timeline_months', 0)
        if timeline:
            content.append(Paragraph(f"Estimated Timeline: {timeline} months", self.styles['BodyText']))
        
        # List milestones
        milestones = [node for node in roadmap.nodes if node.node_type == 'milestone']
        if milestones:
            content.append(Paragraph("Key Milestones:", self.styles['SubSection']))
            
            milestone_data = [['Milestone', 'Timeline', 'Required Skills']]
            for milestone in milestones:
                skills_text = ', '.join(milestone.required_skills[:3]) if milestone.required_skills else 'None specified'
                if len(milestone.required_skills) > 3:
                    skills_text += f" (+{len(milestone.required_skills) - 3} more)"
                
                milestone_data.append([
                    milestone.title,
                    f"{milestone.timeline_months} months" if milestone.timeline_months else "TBD",
                    skills_text
                ])
            
            milestone_table = Table(milestone_data, colWidths=[2*inch, 1*inch, 2.5*inch])
            milestone_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#9b59b6')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lavender),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
            ]))
            
            content.append(milestone_table)
        
        return content
    
    def _create_skill_gap_section(self, report_data: CareerAnalysisReport) -> List:
        """Create skill gap analysis section"""
        content = []
        content.append(Paragraph("Skill Gap Analysis", self.styles['SectionHeader']))
        
        gap_report = report_data.skill_gap_report
        
        # Overall match score
        content.append(Paragraph(f"Overall Match Score: {gap_report.overall_match_score:.1f}%", self.styles['Highlight']))
        
        # Strengths
        if gap_report.strengths:
            content.append(Paragraph("Your Strengths:", self.styles['SubSection']))
            for strength in gap_report.strengths:
                content.append(Paragraph(f"• {strength}", self.styles['BodyText']))
            content.append(Spacer(1, 0.1*inch))
        
        # Skill gaps
        if gap_report.skill_gaps:
            content.append(Paragraph("Skills to Develop:", self.styles['SubSection']))
            
            # Create skill gaps table
            gap_data = [['Skill', 'Current', 'Target', 'Gap', 'Priority', 'Est. Hours']]
            
            for gap in gap_report.skill_gaps[:10]:  # Top 10 gaps
                gap_data.append([
                    gap.skill_name,
                    f"{gap.current_level:.1f}%",
                    f"{gap.target_level:.1f}%",
                    f"{gap.gap_size:.1f}",
                    gap.priority.title(),
                    str(gap.estimated_learning_hours) if gap.estimated_learning_hours else "TBD"
                ])
            
            gap_table = Table(gap_data, colWidths=[1.5*inch, 0.7*inch, 0.7*inch, 0.7*inch, 0.7*inch, 0.7*inch])
            gap_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e74c3c')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.mistyrose),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
            ]))
            
            content.append(gap_table)
        
        return content
    
    def _create_job_compatibility_section(self, report_data: CareerAnalysisReport) -> List:
        """Create job compatibility section"""
        content = []
        content.append(Paragraph("Job Compatibility Analysis", self.styles['SectionHeader']))
        
        job_report = report_data.job_compatibility_report
        
        if job_report.job_matches:
            content.append(Paragraph(f"Analyzed {job_report.total_jobs_analyzed} job opportunities", self.styles['BodyText']))
            
            # Top job matches
            top_matches = job_report.job_matches[:5]  # Top 5 matches
            
            job_data = [['Job Title', 'Company', 'Overall Score', 'Skill Match', 'Recommendation']]
            
            for job in top_matches:
                job_data.append([
                    job.job_title[:30] + "..." if len(job.job_title) > 30 else job.job_title,
                    job.company[:20] + "..." if len(job.company) > 20 else job.company,
                    f"{job.overall_score:.1f}%",
                    f"{job.skill_match_score:.1f}%",
                    job.recommendation.title()
                ])
            
            job_table = Table(job_data, colWidths=[2*inch, 1.5*inch, 1*inch, 1*inch, 1*inch])
            job_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f39c12')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.wheat),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ]))
            
            content.append(job_table)
        
        return content
    
    def _create_progress_section(self, report_data: CareerAnalysisReport) -> List:
        """Create progress tracking section"""
        content = []
        content.append(Paragraph("Progress Tracking", self.styles['SectionHeader']))
        
        progress = report_data.progress_report
        
        content.append(Paragraph(f"Tracking Period: {progress.tracking_period_days} days", self.styles['CustomBodyText']))
        content.append(Paragraph(f"Overall Improvement Score: {progress.overall_improvement_score:.1f}%", self.styles['Highlight']))
        
        # Milestones achieved
        if progress.milestones_achieved:
            content.append(Paragraph("Milestones Achieved:", self.styles['SubSection']))
            for milestone in progress.milestones_achieved:
                content.append(Paragraph(f"• {milestone}", self.styles['CustomBodyText']))
        
        # Skill improvements
        if progress.skill_improvements:
            content.append(Paragraph("Skill Improvements:", self.styles['SubSection']))
            
            improvement_data = [['Skill', 'Previous Score', 'Current Score', 'Improvement']]
            
            for improvement in progress.skill_improvements[:10]:  # Top 10 improvements
                improvement_data.append([
                    improvement.skill_name,
                    f"{improvement.previous_score:.1f}%",
                    f"{improvement.current_score:.1f}%",
                    f"+{improvement.improvement:.1f}%" if improvement.improvement > 0 else f"{improvement.improvement:.1f}%"
                ])
            
            improvement_table = Table(improvement_data, colWidths=[2*inch, 1*inch, 1*inch, 1*inch])
            improvement_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#27ae60')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightgreen),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
            ]))
            
            content.append(improvement_table)
        
        return content
    
    def _create_recommendations_section(self, report_data: CareerAnalysisReport) -> List:
        """Create recommendations section"""
        content = []
        content.append(Paragraph("Recommendations", self.styles['SectionHeader']))
        
        for i, recommendation in enumerate(report_data.recommendations, 1):
            content.append(Paragraph(f"{i}. {recommendation}", self.styles['CustomBodyText']))
        
        return content
    
    def _create_next_steps_section(self, report_data: CareerAnalysisReport) -> List:
        """Create next steps section"""
        content = []
        content.append(Paragraph("Next Steps", self.styles['SectionHeader']))
        
        for i, step in enumerate(report_data.next_steps, 1):
            content.append(Paragraph(f"{i}. {step}", self.styles['CustomBodyText']))
        
        return content
    
    def _create_skill_radar_chart_image(self, radar_data) -> Optional[Image]:
        """Create skill radar chart image using matplotlib"""
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        try:
            # Create figure
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
            
            # Number of variables
            categories = radar_data.categories
            N = len(categories)
            
            # Compute angle for each axis
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # Complete the circle
            
            # Add user scores
            user_scores = radar_data.user_scores + radar_data.user_scores[:1]
            ax.plot(angles, user_scores, 'o-', linewidth=2, label='Your Skills', color='#3498db')
            ax.fill(angles, user_scores, alpha=0.25, color='#3498db')
            
            # Add market average
            market_avg = radar_data.market_average + radar_data.market_average[:1]
            ax.plot(angles, market_avg, 'o--', linewidth=2, label='Market Average', color='#e74c3c')
            
            # Add target scores if available
            if radar_data.target_scores:
                target_scores = radar_data.target_scores + radar_data.target_scores[:1]
                ax.plot(angles, target_scores, 's--', linewidth=2, label='Target Level', color='#2ecc71')
            
            # Add category labels
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            
            # Set y-axis limits
            ax.set_ylim(0, 100)
            ax.set_yticks([20, 40, 60, 80, 100])
            ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
            
            # Add legend
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            
            # Add title
            plt.title('Skill Radar Chart', size=16, fontweight='bold', pad=20)
            
            # Save to bytes
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            plt.close()
            
            # Create ReportLab Image
            return Image(img_buffer, width=6*inch, height=4*inch)
            
        except Exception as e:
            logger.error(f"Error creating radar chart image: {str(e)}")
            return None
    
    async def get_report_file(self, report_id: str) -> Optional[Path]:
        """Get report file path by ID"""
        for file_path in self.storage_path.glob(f"*_{report_id}.*"):
            return file_path
        return None
    
    async def cleanup_expired_reports(self):
        """Clean up expired report files"""
        try:
            current_time = datetime.utcnow()
            for file_path in self.storage_path.glob("career_report_*"):
                # Check file age (7 days)
                file_age = current_time - datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_age.days > 7:
                    file_path.unlink()
                    logger.info(f"Cleaned up expired report: {file_path.name}")
        except Exception as e:
            logger.error(f"Error cleaning up expired reports: {str(e)}")