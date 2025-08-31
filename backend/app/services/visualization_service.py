"""
Visualization service for generating professional charts and visual elements
"""
import json
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import base64
import io
import logging

from ..schemas.analytics import (
    SkillRadarChart, CareerRoadmapVisualization, ChartConfiguration, 
    VisualizationResponse, ChartType, SkillGapReport, JobCompatibilityReport
)
from ..core.exceptions import VisualizationError

logger = logging.getLogger(__name__)


class VisualizationService:
    """Service for generating professional visualizations and charts"""
    
    def __init__(self):
        self.chart_templates = self._load_chart_templates()
        self.color_schemes = self._load_color_schemes()
    
    def generate_skill_radar_chart_data(self, radar_data: SkillRadarChart, config: ChartConfiguration) -> Dict[str, Any]:
        """Generate skill radar chart visualization data"""
        try:
            chart_data = {
                "type": "radar",
                "data": {
                    "labels": radar_data.categories,
                    "datasets": [
                        {
                            "label": "Your Skills",
                            "data": radar_data.user_scores,
                            "backgroundColor": "rgba(54, 162, 235, 0.2)",
                            "borderColor": "rgba(54, 162, 235, 1)",
                            "borderWidth": 2,
                            "pointBackgroundColor": "rgba(54, 162, 235, 1)",
                            "pointBorderColor": "#fff",
                            "pointHoverBackgroundColor": "#fff",
                            "pointHoverBorderColor": "rgba(54, 162, 235, 1)"
                        },
                        {
                            "label": "Market Average",
                            "data": radar_data.market_average,
                            "backgroundColor": "rgba(255, 99, 132, 0.1)",
                            "borderColor": "rgba(255, 99, 132, 1)",
                            "borderWidth": 1,
                            "borderDash": [5, 5],
                            "pointBackgroundColor": "rgba(255, 99, 132, 1)",
                            "pointBorderColor": "#fff",
                            "pointRadius": 3
                        }
                    ]
                },
                "options": {
                    "responsive": True,
                    "maintainAspectRatio": False,
                    "plugins": {
                        "title": {
                            "display": True,
                            "text": config.title,
                            "font": {
                                "size": 18,
                                "weight": "bold"
                            }
                        },
                        "legend": {
                            "position": "top",
                            "labels": {
                                "usePointStyle": True,
                                "padding": 20
                            }
                        },
                        "tooltip": {
                            "callbacks": {
                                "label": "function(context) { return context.dataset.label + ': ' + context.parsed.r.toFixed(1) + '%'; }"
                            }
                        }
                    },
                    "scales": {
                        "r": {
                            "beginAtZero": True,
                            "max": radar_data.max_score,
                            "ticks": {
                                "stepSize": 20,
                                "callback": "function(value) { return value + '%'; }"
                            },
                            "grid": {
                                "color": "rgba(0, 0, 0, 0.1)"
                            },
                            "angleLines": {
                                "color": "rgba(0, 0, 0, 0.1)"
                            },
                            "pointLabels": {
                                "font": {
                                    "size": 12,
                                    "weight": "500"
                                }
                            }
                        }
                    },
                    "elements": {
                        "line": {
                            "tension": 0.1
                        }
                    }
                }
            }
            
            # Add target scores if available
            if radar_data.target_scores:
                chart_data["data"]["datasets"].append({
                    "label": "Target Level",
                    "data": radar_data.target_scores,
                    "backgroundColor": "rgba(75, 192, 192, 0.1)",
                    "borderColor": "rgba(75, 192, 192, 1)",
                    "borderWidth": 2,
                    "borderDash": [10, 5],
                    "pointBackgroundColor": "rgba(75, 192, 192, 1)",
                    "pointBorderColor": "#fff",
                    "pointRadius": 4,
                    "pointStyle": "triangle"
                })
            
            return chart_data
            
        except Exception as e:
            logger.error(f"Error generating radar chart data: {str(e)}")
            raise VisualizationError(f"Failed to generate radar chart: {str(e)}")
    
    def generate_career_roadmap_data(self, roadmap: CareerRoadmapVisualization, config: ChartConfiguration) -> Dict[str, Any]:
        """Generate career roadmap visualization data"""
        try:
            # Convert roadmap to network graph format
            nodes = []
            edges = []
            
            for node in roadmap.nodes:
                node_data = {
                    "id": node.id,
                    "label": node.title,
                    "title": node.description,
                    "x": node.position["x"],
                    "y": node.position["y"],
                    "shape": self._get_node_shape(node.node_type),
                    "color": self._get_node_color(node.node_type, node.completion_status),
                    "size": self._get_node_size(node.node_type),
                    "font": {
                        "size": 14,
                        "color": "#333333"
                    },
                    "borderWidth": 2,
                    "shadow": True
                }
                
                # Add timeline information
                if node.timeline_months:
                    node_data["timeline"] = f"{node.timeline_months} months"
                
                # Add required skills
                if node.required_skills:
                    node_data["skills"] = node.required_skills
                
                nodes.append(node_data)
            
            for edge in roadmap.edges:
                edge_data = {
                    "id": edge.id,
                    "from": edge.source_id,
                    "to": edge.target_id,
                    "arrows": "to",
                    "color": self._get_edge_color(edge.edge_type),
                    "width": self._get_edge_width(edge.difficulty),
                    "dashes": edge.edge_type == "alternative",
                    "smooth": {
                        "type": "continuous",
                        "roundness": 0.2
                    }
                }
                
                # Add edge labels
                if edge.estimated_duration_months:
                    edge_data["label"] = f"{edge.estimated_duration_months}m"
                    edge_data["font"] = {
                        "size": 10,
                        "color": "#666666",
                        "background": "rgba(255, 255, 255, 0.8)"
                    }
                
                edges.append(edge_data)
            
            return {
                "nodes": nodes,
                "edges": edges,
                "options": {
                    "physics": {
                        "enabled": False
                    },
                    "interaction": {
                        "hover": True,
                        "selectConnectedEdges": False
                    },
                    "layout": {
                        "hierarchical": {
                            "enabled": True,
                            "direction": "LR",
                            "sortMethod": "directed",
                            "levelSeparation": 200,
                            "nodeSpacing": 150
                        }
                    }
                },
                "metadata": roadmap.metadata
            }
            
        except Exception as e:
            logger.error(f"Error generating roadmap data: {str(e)}")
            raise VisualizationError(f"Failed to generate roadmap: {str(e)}")
    
    def generate_skill_gap_chart_data(self, gap_report: SkillGapReport, config: ChartConfiguration) -> Dict[str, Any]:
        """Generate skill gap analysis chart data"""
        try:
            # Sort gaps by priority and gap size
            sorted_gaps = sorted(gap_report.skill_gaps, key=lambda x: (
                {"high": 3, "medium": 2, "low": 1}[x.priority], x.gap_size
            ), reverse=True)
            
            skills = [gap.skill_name for gap in sorted_gaps[:10]]  # Top 10 gaps
            current_levels = [gap.current_level for gap in sorted_gaps[:10]]
            target_levels = [gap.target_level for gap in sorted_gaps[:10]]
            gap_sizes = [gap.gap_size for gap in sorted_gaps[:10]]
            
            # Create color mapping based on priority
            colors = []
            for gap in sorted_gaps[:10]:
                if gap.priority == "high":
                    colors.append("rgba(255, 99, 132, 0.8)")
                elif gap.priority == "medium":
                    colors.append("rgba(255, 206, 86, 0.8)")
                else:
                    colors.append("rgba(75, 192, 192, 0.8)")
            
            chart_data = {
                "type": "bar",
                "data": {
                    "labels": skills,
                    "datasets": [
                        {
                            "label": "Current Level",
                            "data": current_levels,
                            "backgroundColor": "rgba(54, 162, 235, 0.6)",
                            "borderColor": "rgba(54, 162, 235, 1)",
                            "borderWidth": 1
                        },
                        {
                            "label": "Target Level",
                            "data": target_levels,
                            "backgroundColor": colors,
                            "borderColor": [color.replace("0.8", "1") for color in colors],
                            "borderWidth": 1
                        }
                    ]
                },
                "options": {
                    "responsive": True,
                    "plugins": {
                        "title": {
                            "display": True,
                            "text": f"Skill Gap Analysis - {gap_report.target_role}",
                            "font": {
                                "size": 16,
                                "weight": "bold"
                            }
                        },
                        "legend": {
                            "position": "top"
                        },
                        "tooltip": {
                            "callbacks": {
                                "afterLabel": "function(context) { const gap = " + str([gap.gap_size for gap in sorted_gaps[:10]]) + "[context.dataIndex]; return 'Gap: ' + gap.toFixed(1) + ' points'; }"
                            }
                        }
                    },
                    "scales": {
                        "y": {
                            "beginAtZero": True,
                            "max": 100,
                            "title": {
                                "display": True,
                                "text": "Skill Level (%)"
                            }
                        },
                        "x": {
                            "title": {
                                "display": True,
                                "text": "Skills"
                            }
                        }
                    }
                }
            }
            
            return chart_data
            
        except Exception as e:
            logger.error(f"Error generating skill gap chart: {str(e)}")
            raise VisualizationError(f"Failed to generate skill gap chart: {str(e)}")
    
    def generate_job_compatibility_chart_data(self, compatibility_report: JobCompatibilityReport, config: ChartConfiguration) -> Dict[str, Any]:
        """Generate job compatibility chart data"""
        try:
            # Get top 10 job matches
            top_jobs = compatibility_report.job_matches[:10]
            
            job_titles = [f"{job.job_title[:20]}..." if len(job.job_title) > 20 else job.job_title for job in top_jobs]
            overall_scores = [job.overall_score for job in top_jobs]
            skill_scores = [job.skill_match_score for job in top_jobs]
            experience_scores = [job.experience_match_score for job in top_jobs]
            
            # Color coding based on recommendation
            colors = []
            for job in top_jobs:
                if job.recommendation == "apply":
                    colors.append("rgba(75, 192, 192, 0.8)")
                elif job.recommendation == "consider":
                    colors.append("rgba(255, 206, 86, 0.8)")
                else:
                    colors.append("rgba(255, 99, 132, 0.8)")
            
            chart_data = {
                "type": "horizontalBar",
                "data": {
                    "labels": job_titles,
                    "datasets": [
                        {
                            "label": "Overall Compatibility",
                            "data": overall_scores,
                            "backgroundColor": colors,
                            "borderColor": [color.replace("0.8", "1") for color in colors],
                            "borderWidth": 1
                        }
                    ]
                },
                "options": {
                    "responsive": True,
                    "indexAxis": "y",
                    "plugins": {
                        "title": {
                            "display": True,
                            "text": "Job Compatibility Scores",
                            "font": {
                                "size": 16,
                                "weight": "bold"
                            }
                        },
                        "legend": {
                            "display": False
                        },
                        "tooltip": {
                            "callbacks": {
                                "afterLabel": "function(context) { const job = " + str([{
                                    "skill_score": job.skill_match_score,
                                    "exp_score": job.experience_match_score,
                                    "recommendation": job.recommendation
                                } for job in top_jobs]) + "[context.dataIndex]; return ['Skill Match: ' + job.skill_score.toFixed(1) + '%', 'Experience Match: ' + job.exp_score.toFixed(1) + '%', 'Recommendation: ' + job.recommendation]; }"
                            }
                        }
                    },
                    "scales": {
                        "x": {
                            "beginAtZero": True,
                            "max": 100,
                            "title": {
                                "display": True,
                                "text": "Compatibility Score (%)"
                            }
                        }
                    }
                }
            }
            
            return chart_data
            
        except Exception as e:
            logger.error(f"Error generating job compatibility chart: {str(e)}")
            raise VisualizationError(f"Failed to generate job compatibility chart: {str(e)}")
    
    def generate_progress_tracking_chart_data(self, progress_data: List[Dict[str, Any]], config: ChartConfiguration) -> Dict[str, Any]:
        """Generate progress tracking chart data"""
        try:
            # Extract data for line chart
            dates = []
            skill_data = {}
            
            # Group progress data by skill
            for entry in progress_data:
                skill_name = entry["skill_name"]
                date = entry["date"]
                score = entry["score"]
                
                if skill_name not in skill_data:
                    skill_data[skill_name] = {"dates": [], "scores": []}
                
                skill_data[skill_name]["dates"].append(date)
                skill_data[skill_name]["scores"].append(score)
                
                if date not in dates:
                    dates.append(date)
            
            dates.sort()
            
            # Create datasets for each skill
            datasets = []
            colors = [
                "rgba(255, 99, 132, 1)", "rgba(54, 162, 235, 1)", "rgba(255, 206, 86, 1)",
                "rgba(75, 192, 192, 1)", "rgba(153, 102, 255, 1)", "rgba(255, 159, 64, 1)"
            ]
            
            for i, (skill_name, data) in enumerate(skill_data.items()):
                color = colors[i % len(colors)]
                datasets.append({
                    "label": skill_name,
                    "data": data["scores"],
                    "borderColor": color,
                    "backgroundColor": color.replace("1)", "0.1)"),
                    "borderWidth": 2,
                    "fill": False,
                    "tension": 0.1,
                    "pointRadius": 4,
                    "pointHoverRadius": 6
                })
            
            chart_data = {
                "type": "line",
                "data": {
                    "labels": dates,
                    "datasets": datasets
                },
                "options": {
                    "responsive": True,
                    "plugins": {
                        "title": {
                            "display": True,
                            "text": "Skill Progress Over Time",
                            "font": {
                                "size": 16,
                                "weight": "bold"
                            }
                        },
                        "legend": {
                            "position": "top"
                        }
                    },
                    "scales": {
                        "y": {
                            "beginAtZero": True,
                            "max": 100,
                            "title": {
                                "display": True,
                                "text": "Skill Level (%)"
                            }
                        },
                        "x": {
                            "title": {
                                "display": True,
                                "text": "Date"
                            }
                        }
                    }
                }
            }
            
            return chart_data
            
        except Exception as e:
            logger.error(f"Error generating progress chart: {str(e)}")
            raise VisualizationError(f"Failed to generate progress chart: {str(e)}")
    
    def create_visualization_response(
        self, 
        chart_type: ChartType, 
        data: Dict[str, Any], 
        config: ChartConfiguration
    ) -> VisualizationResponse:
        """Create visualization response"""
        chart_id = str(uuid.uuid4())
        expires_at = datetime.utcnow() + timedelta(hours=24)  # Charts expire after 24 hours
        
        return VisualizationResponse(
            chart_id=chart_id,
            chart_type=chart_type,
            data=data,
            configuration=config,
            expires_at=expires_at
        )
    
    # Helper methods
    def _get_node_shape(self, node_type: str) -> str:
        """Get node shape based on type"""
        shape_mapping = {
            "current": "dot",
            "target": "star",
            "milestone": "box",
            "alternative": "diamond"
        }
        return shape_mapping.get(node_type, "dot")
    
    def _get_node_color(self, node_type: str, completion_status: Optional[str] = None) -> Dict[str, str]:
        """Get node color based on type and status"""
        if completion_status == "completed":
            return {"background": "#4CAF50", "border": "#45a049"}
        elif completion_status == "in_progress":
            return {"background": "#FF9800", "border": "#F57C00"}
        
        color_mapping = {
            "current": {"background": "#2196F3", "border": "#1976D2"},
            "target": {"background": "#9C27B0", "border": "#7B1FA2"},
            "milestone": {"background": "#FF5722", "border": "#D84315"},
            "alternative": {"background": "#607D8B", "border": "#455A64"}
        }
        return color_mapping.get(node_type, {"background": "#9E9E9E", "border": "#757575"})
    
    def _get_node_size(self, node_type: str) -> int:
        """Get node size based on type"""
        size_mapping = {
            "current": 30,
            "target": 35,
            "milestone": 25,
            "alternative": 20
        }
        return size_mapping.get(node_type, 25)
    
    def _get_edge_color(self, edge_type: str) -> str:
        """Get edge color based on type"""
        color_mapping = {
            "direct": "#2196F3",
            "alternative": "#FF9800",
            "prerequisite": "#4CAF50"
        }
        return color_mapping.get(edge_type, "#9E9E9E")
    
    def _get_edge_width(self, difficulty: float) -> int:
        """Get edge width based on difficulty"""
        if difficulty >= 0.8:
            return 4
        elif difficulty >= 0.6:
            return 3
        elif difficulty >= 0.4:
            return 2
        else:
            return 1
    
    def _load_chart_templates(self) -> Dict[str, Any]:
        """Load chart templates"""
        return {
            "radar": {
                "responsive": True,
                "maintainAspectRatio": False,
                "plugins": {
                    "legend": {"position": "top"}
                }
            },
            "bar": {
                "responsive": True,
                "plugins": {
                    "legend": {"position": "top"}
                }
            },
            "line": {
                "responsive": True,
                "plugins": {
                    "legend": {"position": "top"}
                }
            }
        }
    
    def _load_color_schemes(self) -> Dict[str, List[str]]:
        """Load color schemes"""
        return {
            "default": [
                "rgba(54, 162, 235, 0.8)",
                "rgba(255, 99, 132, 0.8)",
                "rgba(255, 206, 86, 0.8)",
                "rgba(75, 192, 192, 0.8)",
                "rgba(153, 102, 255, 0.8)",
                "rgba(255, 159, 64, 0.8)"
            ],
            "professional": [
                "rgba(63, 81, 181, 0.8)",
                "rgba(33, 150, 243, 0.8)",
                "rgba(0, 188, 212, 0.8)",
                "rgba(0, 150, 136, 0.8)",
                "rgba(76, 175, 80, 0.8)",
                "rgba(139, 195, 74, 0.8)"
            ],
            "warm": [
                "rgba(244, 67, 54, 0.8)",
                "rgba(233, 30, 99, 0.8)",
                "rgba(156, 39, 176, 0.8)",
                "rgba(103, 58, 183, 0.8)",
                "rgba(63, 81, 181, 0.8)",
                "rgba(33, 150, 243, 0.8)"
            ]
        }