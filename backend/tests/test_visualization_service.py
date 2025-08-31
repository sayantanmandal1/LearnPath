"""
Tests for visualization service functionality
"""
import pytest
from datetime import datetime
from unittest.mock import Mock

from app.services.visualization_service import VisualizationService
from app.schemas.analytics import (
    SkillRadarChart, CareerRoadmapVisualization, CareerRoadmapNode, CareerRoadmapEdge,
    SkillGapReport, SkillGapAnalysis, JobCompatibilityReport, JobCompatibilityScore,
    ChartConfiguration, ChartType, VisualizationResponse
)
from app.core.exceptions import VisualizationError


@pytest.fixture
def viz_service():
    """Visualization service instance"""
    return VisualizationService()


@pytest.fixture
def sample_radar_chart():
    """Sample radar chart data"""
    return SkillRadarChart(
        user_id="test-user-123",
        categories=["Programming", "Frameworks", "Databases", "Cloud", "Soft Skills", "Tools"],
        user_scores=[85.0, 70.0, 60.0, 50.0, 80.0, 75.0],
        market_average=[65.0, 65.0, 65.0, 65.0, 65.0, 65.0],
        target_scores=[90.0, 85.0, 75.0, 70.0, 85.0, 80.0],
        max_score=100.0
    )


@pytest.fixture
def sample_roadmap():
    """Sample career roadmap data"""
    nodes = [
        CareerRoadmapNode(
            id="current",
            title="Software Developer",
            description="Current position",
            position={"x": 0, "y": 0},
            node_type="current",
            timeline_months=0,
            completion_status="completed"
        ),
        CareerRoadmapNode(
            id="milestone_1",
            title="Senior Developer",
            description="First milestone",
            position={"x": 200, "y": 0},
            node_type="milestone",
            timeline_months=12,
            required_skills=["Python", "React"],
            completion_status="not_started"
        ),
        CareerRoadmapNode(
            id="target",
            title="Tech Lead",
            description="Target position",
            position={"x": 400, "y": 0},
            node_type="target",
            timeline_months=24,
            completion_status="not_started"
        )
    ]
    
    edges = [
        CareerRoadmapEdge(
            id="edge_1",
            source_id="current",
            target_id="milestone_1",
            edge_type="direct",
            difficulty=0.7,
            estimated_duration_months=12
        ),
        CareerRoadmapEdge(
            id="edge_2",
            source_id="milestone_1",
            target_id="target",
            edge_type="direct",
            difficulty=0.8,
            estimated_duration_months=12
        )
    ]
    
    return CareerRoadmapVisualization(
        user_id="test-user-123",
        nodes=nodes,
        edges=edges,
        metadata={"target_role": "Tech Lead", "total_timeline_months": 24}
    )


@pytest.fixture
def sample_skill_gap_report():
    """Sample skill gap report"""
    skill_gaps = [
        SkillGapAnalysis(
            skill_name="Python",
            current_level=70.0,
            target_level=90.0,
            gap_size=20.0,
            priority="high",
            estimated_learning_hours=40
        ),
        SkillGapAnalysis(
            skill_name="React",
            current_level=60.0,
            target_level=85.0,
            gap_size=25.0,
            priority="high",
            estimated_learning_hours=50
        ),
        SkillGapAnalysis(
            skill_name="Docker",
            current_level=30.0,
            target_level=70.0,
            gap_size=40.0,
            priority="medium",
            estimated_learning_hours=80
        )
    ]
    
    return SkillGapReport(
        user_id="test-user-123",
        target_role="Senior Developer",
        overall_match_score=75.0,
        skill_gaps=skill_gaps,
        strengths=["JavaScript", "HTML", "CSS"],
        total_learning_hours=170,
        priority_skills=["Python", "React"]
    )


@pytest.fixture
def sample_job_compatibility_report():
    """Sample job compatibility report"""
    job_matches = [
        JobCompatibilityScore(
            job_id="job-1",
            job_title="Senior Python Developer",
            company="Tech Corp",
            overall_score=85.0,
            skill_match_score=80.0,
            experience_match_score=90.0,
            matched_skills=["Python", "JavaScript", "React"],
            missing_skills=["Docker", "Kubernetes"],
            recommendation="apply"
        ),
        JobCompatibilityScore(
            job_id="job-2",
            job_title="Full Stack Developer",
            company="Startup Inc",
            overall_score=70.0,
            skill_match_score=65.0,
            experience_match_score=75.0,
            matched_skills=["JavaScript", "React"],
            missing_skills=["Python", "AWS", "Docker"],
            recommendation="consider"
        ),
        JobCompatibilityScore(
            job_id="job-3",
            job_title="DevOps Engineer",
            company="Cloud Solutions",
            overall_score=45.0,
            skill_match_score=40.0,
            experience_match_score=50.0,
            matched_skills=["Python"],
            missing_skills=["AWS", "Docker", "Kubernetes", "Terraform"],
            recommendation="improve_first"
        )
    ]
    
    return JobCompatibilityReport(
        user_id="test-user-123",
        job_matches=job_matches,
        filters_applied={"location": "San Francisco"},
        total_jobs_analyzed=10
    )


@pytest.fixture
def sample_chart_config():
    """Sample chart configuration"""
    return ChartConfiguration(
        chart_type=ChartType.RADAR,
        title="Skill Analysis",
        width=800,
        height=600,
        color_scheme="professional",
        interactive=True,
        export_format="svg"
    )


class TestVisualizationService:
    """Test cases for VisualizationService"""
    
    def test_generate_skill_radar_chart_data(self, viz_service, sample_radar_chart, sample_chart_config):
        """Test skill radar chart data generation"""
        result = viz_service.generate_skill_radar_chart_data(sample_radar_chart, sample_chart_config)
        
        assert result["type"] == "radar"
        assert "data" in result
        assert "options" in result
        
        # Check data structure
        data = result["data"]
        assert "labels" in data
        assert "datasets" in data
        assert len(data["labels"]) == 6  # Number of categories
        assert len(data["datasets"]) == 3  # User, market average, target
        
        # Check user scores dataset
        user_dataset = data["datasets"][0]
        assert user_dataset["label"] == "Your Skills"
        assert len(user_dataset["data"]) == 6
        assert user_dataset["data"] == sample_radar_chart.user_scores
        
        # Check market average dataset
        market_dataset = data["datasets"][1]
        assert market_dataset["label"] == "Market Average"
        assert len(market_dataset["data"]) == 6
        assert market_dataset["data"] == sample_radar_chart.market_average
        
        # Check target scores dataset
        target_dataset = data["datasets"][2]
        assert target_dataset["label"] == "Target Level"
        assert len(target_dataset["data"]) == 6
        assert target_dataset["data"] == sample_radar_chart.target_scores
        
        # Check options
        options = result["options"]
        assert options["responsive"] is True
        assert options["plugins"]["title"]["text"] == sample_chart_config.title
    
    def test_generate_skill_radar_chart_data_without_target(self, viz_service, sample_chart_config):
        """Test radar chart generation without target scores"""
        radar_chart = SkillRadarChart(
            user_id="test-user-123",
            categories=["Programming", "Frameworks"],
            user_scores=[85.0, 70.0],
            market_average=[65.0, 65.0],
            target_scores=None,  # No target scores
            max_score=100.0
        )
        
        result = viz_service.generate_skill_radar_chart_data(radar_chart, sample_chart_config)
        
        # Should only have 2 datasets (user and market average)
        assert len(result["data"]["datasets"]) == 2
        assert result["data"]["datasets"][0]["label"] == "Your Skills"
        assert result["data"]["datasets"][1]["label"] == "Market Average"
    
    def test_generate_career_roadmap_data(self, viz_service, sample_roadmap, sample_chart_config):
        """Test career roadmap data generation"""
        result = viz_service.generate_career_roadmap_data(sample_roadmap, sample_chart_config)
        
        assert "nodes" in result
        assert "edges" in result
        assert "options" in result
        assert "metadata" in result
        
        # Check nodes
        nodes = result["nodes"]
        assert len(nodes) == 3  # current, milestone, target
        
        current_node = nodes[0]
        assert current_node["id"] == "current"
        assert current_node["label"] == "Software Developer"
        assert current_node["shape"] == "dot"  # Current node shape
        
        milestone_node = nodes[1]
        assert milestone_node["id"] == "milestone_1"
        assert milestone_node["label"] == "Senior Developer"
        assert milestone_node["shape"] == "box"  # Milestone node shape
        
        target_node = nodes[2]
        assert target_node["id"] == "target"
        assert target_node["label"] == "Tech Lead"
        assert target_node["shape"] == "star"  # Target node shape
        
        # Check edges
        edges = result["edges"]
        assert len(edges) == 2
        
        first_edge = edges[0]
        assert first_edge["from"] == "current"
        assert first_edge["to"] == "milestone_1"
        assert first_edge["arrows"] == "to"
        
        # Check metadata
        assert result["metadata"] == sample_roadmap.metadata
    
    def test_generate_skill_gap_chart_data(self, viz_service, sample_skill_gap_report, sample_chart_config):
        """Test skill gap chart data generation"""
        result = viz_service.generate_skill_gap_chart_data(sample_skill_gap_report, sample_chart_config)
        
        assert result["type"] == "bar"
        assert "data" in result
        assert "options" in result
        
        # Check data structure
        data = result["data"]
        assert "labels" in data
        assert "datasets" in data
        
        # Should have skills as labels (top 10 gaps)
        labels = data["labels"]
        assert len(labels) <= 10
        assert "Python" in labels
        assert "React" in labels
        assert "Docker" in labels
        
        # Should have 2 datasets (current and target levels)
        datasets = data["datasets"]
        assert len(datasets) == 2
        assert datasets[0]["label"] == "Current Level"
        assert datasets[1]["label"] == "Target Level"
        
        # Check options
        options = result["options"]
        assert options["responsive"] is True
        assert "Skill Gap Analysis" in options["plugins"]["title"]["text"]
        assert sample_skill_gap_report.target_role in options["plugins"]["title"]["text"]
    
    def test_generate_job_compatibility_chart_data(self, viz_service, sample_job_compatibility_report, sample_chart_config):
        """Test job compatibility chart data generation"""
        result = viz_service.generate_job_compatibility_chart_data(sample_job_compatibility_report, sample_chart_config)
        
        assert result["type"] == "horizontalBar"
        assert "data" in result
        assert "options" in result
        
        # Check data structure
        data = result["data"]
        assert "labels" in data
        assert "datasets" in data
        
        # Should have job titles as labels (top 10)
        labels = data["labels"]
        assert len(labels) <= 10
        assert len(labels) == 3  # We have 3 sample jobs
        
        # Should have 1 dataset (overall compatibility)
        datasets = data["datasets"]
        assert len(datasets) == 1
        assert datasets[0]["label"] == "Overall Compatibility"
        
        # Check color coding based on recommendations
        colors = datasets[0]["backgroundColor"]
        assert len(colors) == 3
        # First job (apply) should be green-ish, second (consider) yellow-ish, third (improve_first) red-ish
        
        # Check options
        options = result["options"]
        assert options["responsive"] is True
        assert options["indexAxis"] == "y"  # Horizontal bar chart
        assert options["plugins"]["title"]["text"] == "Job Compatibility Scores"
    
    def test_generate_progress_tracking_chart_data(self, viz_service, sample_chart_config):
        """Test progress tracking chart data generation"""
        progress_data = [
            {"skill_name": "Python", "date": "2024-01-01", "score": 70.0},
            {"skill_name": "Python", "date": "2024-02-01", "score": 75.0},
            {"skill_name": "Python", "date": "2024-03-01", "score": 80.0},
            {"skill_name": "React", "date": "2024-01-01", "score": 60.0},
            {"skill_name": "React", "date": "2024-02-01", "score": 65.0},
            {"skill_name": "React", "date": "2024-03-01", "score": 70.0},
        ]
        
        result = viz_service.generate_progress_tracking_chart_data(progress_data, sample_chart_config)
        
        assert result["type"] == "line"
        assert "data" in result
        assert "options" in result
        
        # Check data structure
        data = result["data"]
        assert "labels" in data  # Dates
        assert "datasets" in data  # Skills
        
        # Should have datasets for each skill
        datasets = data["datasets"]
        assert len(datasets) == 2  # Python and React
        
        python_dataset = next(ds for ds in datasets if ds["label"] == "Python")
        assert len(python_dataset["data"]) == 3  # 3 data points
        assert python_dataset["data"] == [70.0, 75.0, 80.0]
        
        react_dataset = next(ds for ds in datasets if ds["label"] == "React")
        assert len(react_dataset["data"]) == 3
        assert react_dataset["data"] == [60.0, 65.0, 70.0]
        
        # Check options
        options = result["options"]
        assert options["responsive"] is True
        assert options["plugins"]["title"]["text"] == "Skill Progress Over Time"
    
    def test_create_visualization_response(self, viz_service, sample_chart_config):
        """Test visualization response creation"""
        chart_data = {"type": "radar", "data": {}, "options": {}}
        
        result = viz_service.create_visualization_response(
            ChartType.RADAR, chart_data, sample_chart_config
        )
        
        assert isinstance(result, VisualizationResponse)
        assert result.chart_type == ChartType.RADAR
        assert result.data == chart_data
        assert result.configuration == sample_chart_config
        assert result.chart_id is not None
        assert result.generated_at is not None
        assert result.expires_at is not None
    
    def test_get_node_shape(self, viz_service):
        """Test node shape mapping"""
        assert viz_service._get_node_shape("current") == "dot"
        assert viz_service._get_node_shape("target") == "star"
        assert viz_service._get_node_shape("milestone") == "box"
        assert viz_service._get_node_shape("alternative") == "diamond"
        assert viz_service._get_node_shape("unknown") == "dot"  # Default
    
    def test_get_node_color(self, viz_service):
        """Test node color mapping"""
        # Test completion status colors
        completed_color = viz_service._get_node_color("milestone", "completed")
        assert completed_color["background"] == "#4CAF50"  # Green
        
        in_progress_color = viz_service._get_node_color("milestone", "in_progress")
        assert in_progress_color["background"] == "#FF9800"  # Orange
        
        # Test node type colors
        current_color = viz_service._get_node_color("current")
        assert current_color["background"] == "#2196F3"  # Blue
        
        target_color = viz_service._get_node_color("target")
        assert target_color["background"] == "#9C27B0"  # Purple
    
    def test_get_node_size(self, viz_service):
        """Test node size mapping"""
        assert viz_service._get_node_size("current") == 30
        assert viz_service._get_node_size("target") == 35
        assert viz_service._get_node_size("milestone") == 25
        assert viz_service._get_node_size("alternative") == 20
        assert viz_service._get_node_size("unknown") == 25  # Default
    
    def test_get_edge_color(self, viz_service):
        """Test edge color mapping"""
        assert viz_service._get_edge_color("direct") == "#2196F3"
        assert viz_service._get_edge_color("alternative") == "#FF9800"
        assert viz_service._get_edge_color("prerequisite") == "#4CAF50"
        assert viz_service._get_edge_color("unknown") == "#9E9E9E"  # Default
    
    def test_get_edge_width(self, viz_service):
        """Test edge width based on difficulty"""
        assert viz_service._get_edge_width(0.9) == 4  # High difficulty
        assert viz_service._get_edge_width(0.7) == 3  # Medium-high difficulty
        assert viz_service._get_edge_width(0.5) == 2  # Medium difficulty
        assert viz_service._get_edge_width(0.2) == 1  # Low difficulty
    
    def test_error_handling(self, viz_service, sample_chart_config):
        """Test error handling in visualization service"""
        # Test with invalid radar chart data
        invalid_radar = Mock()
        invalid_radar.categories = None  # This should cause an error
        
        with pytest.raises(VisualizationError):
            viz_service.generate_skill_radar_chart_data(invalid_radar, sample_chart_config)
    
    def test_load_chart_templates(self, viz_service):
        """Test chart templates loading"""
        templates = viz_service._load_chart_templates()
        
        assert "radar" in templates
        assert "bar" in templates
        assert "line" in templates
        
        # Check template structure
        radar_template = templates["radar"]
        assert radar_template["responsive"] is True
        assert "plugins" in radar_template
    
    def test_load_color_schemes(self, viz_service):
        """Test color schemes loading"""
        color_schemes = viz_service._load_color_schemes()
        
        assert "default" in color_schemes
        assert "professional" in color_schemes
        assert "warm" in color_schemes
        
        # Check color scheme structure
        default_colors = color_schemes["default"]
        assert isinstance(default_colors, list)
        assert len(default_colors) >= 6  # Should have multiple colors
        assert all(color.startswith("rgba(") for color in default_colors)