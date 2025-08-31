"""
Market trend analyzer for job market data analysis and predictions
"""
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.cluster import DBSCAN
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_

from app.models.job import JobPosting, JobSkill
from app.models.skill import Skill
from app.repositories.job import JobRepository

logger = logging.getLogger(__name__)


@dataclass
class TrendData:
    """Data structure for trend analysis"""
    dates: List[datetime]
    values: List[float]
    skill_name: str
    metric_type: str  # 'demand', 'salary', 'growth'


@dataclass
class SalaryPrediction:
    """Salary prediction result"""
    skill_name: str
    location: Optional[str]
    experience_level: Optional[str]
    predicted_salary: float
    confidence_interval: Tuple[float, float]
    model_accuracy: float
    sample_size: int


@dataclass
class EmergingSkill:
    """Emerging skill detection result"""
    skill_name: str
    growth_rate: float
    current_demand: int
    trend_score: float
    confidence: float
    related_skills: List[str]


class MarketTrendAnalyzer:
    """Advanced market trend analysis and prediction system"""
    
    def __init__(self):
        self.job_repository = JobRepository()
        self.scaler = StandardScaler()
        
        # Model cache
        self._salary_models = {}
        self._trend_models = {}
    
    async def analyze_skill_demand_trends(
        self,
        db: AsyncSession,
        skill_names: Optional[List[str]] = None,
        days: int = 180,
        time_window: int = 7  # Weekly aggregation
    ) -> List[Dict[str, Any]]:
        """
        Analyze demand trends for skills over time
        
        Args:
            db: Database session
            skill_names: Specific skills to analyze (None for top skills)
            days: Number of days to analyze
            time_window: Aggregation window in days
            
        Returns:
            List of trend analysis results
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Get skills to analyze
        if skill_names is None:
            skill_names = await self._get_top_skills(db, days, limit=50)
        
        trend_results = []
        
        for skill_name in skill_names:
            try:
                # Get time series data for skill demand
                demand_data = await self._get_skill_demand_timeseries(
                    db, skill_name, cutoff_date, time_window
                )
                
                if len(demand_data) < 3:  # Need minimum data points
                    continue
                
                # Perform trend analysis
                trend_analysis = self._analyze_time_series(demand_data, skill_name)
                
                # Calculate additional metrics
                trend_analysis.update({
                    'skill_name': skill_name,
                    'analysis_period_days': days,
                    'data_points': len(demand_data),
                    'current_demand': demand_data[-1]['value'] if demand_data else 0,
                    'peak_demand': max(point['value'] for point in demand_data),
                    'average_demand': np.mean([point['value'] for point in demand_data])
                })
                
                trend_results.append(trend_analysis)
                
            except Exception as e:
                logger.error(f"Failed to analyze trend for {skill_name}: {str(e)}")
                continue
        
        # Sort by trend strength
        trend_results.sort(key=lambda x: abs(x.get('trend_slope', 0)), reverse=True)
        
        return trend_results
    
    async def _get_skill_demand_timeseries(
        self,
        db: AsyncSession,
        skill_name: str,
        start_date: datetime,
        window_days: int
    ) -> List[Dict[str, Any]]:
        """Get time series data for skill demand"""
        
        # Query job postings with this skill over time
        query = select(
            func.date_trunc('week', JobPosting.posted_date).label('week'),
            func.count(JobSkill.id).label('demand_count')
        ).select_from(
            JobPosting.__table__.join(JobSkill.__table__).join(Skill.__table__)
        ).where(
            and_(
                Skill.name == skill_name,
                JobPosting.posted_date >= start_date,
                JobPosting.is_active == True
            )
        ).group_by(
            func.date_trunc('week', JobPosting.posted_date)
        ).order_by(
            func.date_trunc('week', JobPosting.posted_date)
        )
        
        result = await db.execute(query)
        data = result.fetchall()
        
        return [
            {
                'date': row.week,
                'value': float(row.demand_count)
            }
            for row in data
        ]
    
    def _analyze_time_series(
        self,
        data: List[Dict[str, Any]],
        skill_name: str
    ) -> Dict[str, Any]:
        """Analyze time series data for trends"""
        
        if len(data) < 2:
            return {'trend_slope': 0, 'trend_direction': 'stable', 'confidence': 0}
        
        # Convert to numpy arrays
        dates = [point['date'] for point in data]
        values = [point['value'] for point in data]
        
        # Convert dates to numeric (days since first date)
        date_nums = [(date - dates[0]).days for date in dates]
        
        # Linear regression for trend
        X = np.array(date_nums).reshape(-1, 1)
        y = np.array(values)
        
        model = LinearRegression()
        model.fit(X, y)
        
        trend_slope = model.coef_[0]
        r2 = model.score(X, y)
        
        # Determine trend direction
        if abs(trend_slope) < 0.1:
            trend_direction = 'stable'
        elif trend_slope > 0:
            trend_direction = 'growing'
        else:
            trend_direction = 'declining'
        
        # Calculate volatility
        volatility = np.std(values) / np.mean(values) if np.mean(values) > 0 else 0
        
        # Seasonal analysis (if enough data)
        seasonality = self._detect_seasonality(values) if len(values) > 12 else 0
        
        return {
            'trend_slope': float(trend_slope),
            'trend_direction': trend_direction,
            'confidence': float(r2),
            'volatility': float(volatility),
            'seasonality_strength': float(seasonality),
            'growth_rate_weekly': float(trend_slope / np.mean(values)) if np.mean(values) > 0 else 0
        }
    
    def _detect_seasonality(self, values: List[float]) -> float:
        """Detect seasonality in time series data"""
        if len(values) < 12:
            return 0
        
        # Simple autocorrelation at lag 4 (monthly) and lag 12 (quarterly)
        values_array = np.array(values)
        
        # Calculate autocorrelation
        def autocorr(x, lag):
            if len(x) <= lag:
                return 0
            return np.corrcoef(x[:-lag], x[lag:])[0, 1]
        
        monthly_corr = autocorr(values_array, 4)
        quarterly_corr = autocorr(values_array, 12)
        
        return max(abs(monthly_corr), abs(quarterly_corr))
    
    async def predict_salaries(
        self,
        db: AsyncSession,
        skill_names: List[str],
        location: Optional[str] = None,
        experience_level: Optional[str] = None,
        days: int = 365
    ) -> List[SalaryPrediction]:
        """
        Predict salaries for skills using regression models
        
        Args:
            db: Database session
            skill_names: Skills to predict salaries for
            location: Target location
            experience_level: Target experience level
            days: Historical data period
            
        Returns:
            List of salary predictions
        """
        predictions = []
        
        for skill_name in skill_names:
            try:
                prediction = await self._predict_skill_salary(
                    db, skill_name, location, experience_level, days
                )
                if prediction:
                    predictions.append(prediction)
            except Exception as e:
                logger.error(f"Failed to predict salary for {skill_name}: {str(e)}")
                continue
        
        return predictions
    
    async def _predict_skill_salary(
        self,
        db: AsyncSession,
        skill_name: str,
        location: Optional[str],
        experience_level: Optional[str],
        days: int
    ) -> Optional[SalaryPrediction]:
        """Predict salary for a specific skill"""
        
        # Get training data
        training_data = await self._get_salary_training_data(
            db, skill_name, location, experience_level, days
        )
        
        if len(training_data) < 10:  # Need minimum samples
            logger.warning(f"Insufficient data for salary prediction: {skill_name}")
            return None
        
        # Prepare features and target
        df = pd.DataFrame(training_data)
        
        # Feature engineering
        features = self._engineer_salary_features(df)
        target = df['salary'].values
        
        if len(features) == 0:
            return None
        
        # Train model
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42
        )
        
        # Try multiple models and select best
        models = {
            'linear': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        
        best_model = None
        best_score = -np.inf
        
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                score = model.score(X_test, y_test)
                
                if score > best_score:
                    best_score = score
                    best_model = model
            except Exception as e:
                logger.warning(f"Model {name} failed: {str(e)}")
                continue
        
        if best_model is None:
            return None
        
        # Make prediction
        # Use median feature values for prediction
        median_features = np.median(features, axis=0).reshape(1, -1)
        predicted_salary = best_model.predict(median_features)[0]
        
        # Calculate confidence interval
        predictions = best_model.predict(X_test)
        residuals = y_test - predictions
        std_residual = np.std(residuals)
        
        confidence_interval = (
            predicted_salary - 1.96 * std_residual,
            predicted_salary + 1.96 * std_residual
        )
        
        return SalaryPrediction(
            skill_name=skill_name,
            location=location,
            experience_level=experience_level,
            predicted_salary=float(predicted_salary),
            confidence_interval=confidence_interval,
            model_accuracy=float(best_score),
            sample_size=len(training_data)
        )
    
    async def _get_salary_training_data(
        self,
        db: AsyncSession,
        skill_name: str,
        location: Optional[str],
        experience_level: Optional[str],
        days: int
    ) -> List[Dict[str, Any]]:
        """Get training data for salary prediction"""
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Build query
        query = select(
            JobPosting.salary_min,
            JobPosting.salary_max,
            JobPosting.salary_period,
            JobPosting.location,
            JobPosting.experience_level,
            JobPosting.remote_type,
            JobPosting.company,
            JobPosting.posted_date
        ).select_from(
            JobPosting.__table__.join(JobSkill.__table__).join(Skill.__table__)
        ).where(
            and_(
                Skill.name == skill_name,
                JobPosting.posted_date >= cutoff_date,
                or_(
                    JobPosting.salary_min.isnot(None),
                    JobPosting.salary_max.isnot(None)
                )
            )
        )
        
        # Add filters
        if location:
            query = query.where(JobPosting.location.ilike(f"%{location}%"))
        
        if experience_level:
            query = query.where(JobPosting.experience_level == experience_level)
        
        result = await db.execute(query)
        data = result.fetchall()
        
        # Process salary data
        training_data = []
        for row in data:
            # Calculate average salary
            if row.salary_min and row.salary_max:
                avg_salary = (row.salary_min + row.salary_max) / 2
            else:
                avg_salary = row.salary_min or row.salary_max
            
            if not avg_salary:
                continue
            
            # Normalize to yearly salary
            if row.salary_period == 'hourly':
                avg_salary *= 2080  # 40 hours/week * 52 weeks
            elif row.salary_period == 'monthly':
                avg_salary *= 12
            
            training_data.append({
                'salary': avg_salary,
                'location': row.location or 'Unknown',
                'experience_level': row.experience_level or 'Unknown',
                'remote_type': row.remote_type or 'Unknown',
                'company': row.company,
                'posted_date': row.posted_date
            })
        
        return training_data
    
    def _engineer_salary_features(self, df: pd.DataFrame) -> np.ndarray:
        """Engineer features for salary prediction"""
        
        features = []
        
        # Experience level encoding
        exp_mapping = {'entry': 1, 'mid': 2, 'senior': 3, 'executive': 4}
        exp_features = df['experience_level'].map(
            lambda x: exp_mapping.get(x, 2)
        ).values
        features.append(exp_features)
        
        # Remote type encoding
        remote_mapping = {'remote': 2, 'hybrid': 1, 'onsite': 0}
        remote_features = df['remote_type'].map(
            lambda x: remote_mapping.get(x, 0)
        ).values
        features.append(remote_features)
        
        # Time features (month, year)
        df['month'] = pd.to_datetime(df['posted_date']).dt.month
        df['year'] = pd.to_datetime(df['posted_date']).dt.year
        features.append(df['month'].values)
        features.append(df['year'].values)
        
        # Location features (simplified - could be enhanced with geo-encoding)
        # For now, just use a simple hash
        location_hash = df['location'].apply(lambda x: hash(x) % 1000).values
        features.append(location_hash)
        
        if len(features) == 0:
            return np.array([])
        
        # Stack features
        feature_matrix = np.column_stack(features)
        
        # Scale features
        return self.scaler.fit_transform(feature_matrix)
    
    async def detect_emerging_skills(
        self,
        db: AsyncSession,
        days: int = 90,
        min_growth_rate: float = 0.1,
        min_sample_size: int = 5
    ) -> List[EmergingSkill]:
        """
        Detect emerging skills using anomaly detection
        
        Args:
            db: Database session
            days: Analysis period
            min_growth_rate: Minimum growth rate to consider
            min_sample_size: Minimum number of job postings
            
        Returns:
            List of emerging skills
        """
        # Get skill trends
        trends = await self.analyze_skill_demand_trends(db, days=days)
        
        # Filter for potential emerging skills
        candidates = [
            trend for trend in trends
            if (trend.get('growth_rate_weekly', 0) > min_growth_rate and
                trend.get('data_points', 0) >= min_sample_size and
                trend.get('trend_direction') == 'growing')
        ]
        
        if not candidates:
            return []
        
        # Extract features for anomaly detection
        features = []
        for trend in candidates:
            features.append([
                trend.get('growth_rate_weekly', 0),
                trend.get('current_demand', 0),
                trend.get('volatility', 0),
                trend.get('confidence', 0)
            ])
        
        features_array = np.array(features)
        
        # Apply DBSCAN for anomaly detection
        # Skills with high growth and low volatility are potential emerging skills
        clustering = DBSCAN(eps=0.3, min_samples=2)
        clusters = clustering.fit_predict(features_array)
        
        emerging_skills = []
        
        for i, trend in enumerate(candidates):
            # Calculate trend score
            trend_score = (
                trend.get('growth_rate_weekly', 0) * 0.4 +
                trend.get('confidence', 0) * 0.3 +
                (1 - trend.get('volatility', 1)) * 0.3
            )
            
            # Get related skills
            related_skills = await self._find_related_skills(
                db, trend['skill_name']
            )
            
            emerging_skill = EmergingSkill(
                skill_name=trend['skill_name'],
                growth_rate=trend.get('growth_rate_weekly', 0),
                current_demand=int(trend.get('current_demand', 0)),
                trend_score=trend_score,
                confidence=trend.get('confidence', 0),
                related_skills=related_skills
            )
            
            emerging_skills.append(emerging_skill)
        
        # Sort by trend score
        emerging_skills.sort(key=lambda x: x.trend_score, reverse=True)
        
        return emerging_skills[:20]  # Return top 20
    
    async def _find_related_skills(
        self,
        db: AsyncSession,
        skill_name: str,
        limit: int = 5
    ) -> List[str]:
        """Find skills that commonly appear with the given skill"""
        
        # Get job IDs that require this skill
        job_ids_query = select(JobSkill.job_posting_id).where(
            JobSkill.skill_id.in_(
                select(Skill.id).where(Skill.name == skill_name)
            )
        )
        
        result = await db.execute(job_ids_query)
        job_ids = [row[0] for row in result.fetchall()]
        
        if not job_ids:
            return []
        
        # Find other skills in these jobs
        related_query = select(
            Skill.name,
            func.count(JobSkill.id).label('co_occurrence')
        ).select_from(
            JobSkill.__table__.join(Skill.__table__)
        ).where(
            and_(
                JobSkill.job_posting_id.in_(job_ids),
                Skill.name != skill_name
            )
        ).group_by(
            Skill.id, Skill.name
        ).order_by(
            func.count(JobSkill.id).desc()
        ).limit(limit)
        
        result = await db.execute(related_query)
        related = result.fetchall()
        
        return [row.name for row in related]
    
    async def _get_top_skills(
        self,
        db: AsyncSession,
        days: int,
        limit: int = 50
    ) -> List[str]:
        """Get top skills by demand in the specified period"""
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        query = select(
            Skill.name,
            func.count(JobSkill.id).label('demand_count')
        ).select_from(
            JobSkill.__table__.join(Skill.__table__).join(JobPosting.__table__)
        ).where(
            JobPosting.posted_date >= cutoff_date
        ).group_by(
            Skill.id, Skill.name
        ).order_by(
            func.count(JobSkill.id).desc()
        ).limit(limit)
        
        result = await db.execute(query)
        skills = result.fetchall()
        
        return [row.name for row in skills]
    
    async def generate_market_report(
        self,
        db: AsyncSession,
        days: int = 90
    ) -> Dict[str, Any]:
        """
        Generate comprehensive market analysis report
        
        Args:
            db: Database session
            days: Analysis period
            
        Returns:
            Comprehensive market report
        """
        logger.info(f"Generating market report for {days} days")
        
        # Get skill trends
        skill_trends = await self.analyze_skill_demand_trends(db, days=days)
        
        # Get emerging skills
        emerging_skills = await self.detect_emerging_skills(db, days=days)
        
        # Get salary predictions for top skills
        top_skills = [trend['skill_name'] for trend in skill_trends[:20]]
        salary_predictions = await self.predict_salaries(db, top_skills, days=days)
        
        # Calculate market statistics
        total_jobs = await self._count_total_jobs(db, days)
        total_skills = len(skill_trends)
        
        # Growth analysis
        growing_skills = [
            trend for trend in skill_trends
            if trend.get('trend_direction') == 'growing'
        ]
        declining_skills = [
            trend for trend in skill_trends
            if trend.get('trend_direction') == 'declining'
        ]
        
        return {
            'report_generated': datetime.utcnow(),
            'analysis_period_days': days,
            'market_overview': {
                'total_jobs_analyzed': total_jobs,
                'total_skills_tracked': total_skills,
                'growing_skills_count': len(growing_skills),
                'declining_skills_count': len(declining_skills),
                'emerging_skills_count': len(emerging_skills)
            },
            'skill_trends': skill_trends[:50],
            'emerging_skills': [
                {
                    'skill_name': skill.skill_name,
                    'growth_rate': skill.growth_rate,
                    'current_demand': skill.current_demand,
                    'trend_score': skill.trend_score,
                    'confidence': skill.confidence,
                    'related_skills': skill.related_skills
                }
                for skill in emerging_skills
            ],
            'salary_predictions': [
                {
                    'skill_name': pred.skill_name,
                    'predicted_salary': pred.predicted_salary,
                    'confidence_interval': pred.confidence_interval,
                    'model_accuracy': pred.model_accuracy,
                    'sample_size': pred.sample_size
                }
                for pred in salary_predictions
            ],
            'top_growing_skills': growing_skills[:10],
            'top_declining_skills': declining_skills[:10]
        }
    
    async def _count_total_jobs(self, db: AsyncSession, days: int) -> int:
        """Count total jobs in analysis period"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        query = select(func.count(JobPosting.id)).where(
            JobPosting.posted_date >= cutoff_date
        )
        
        result = await db.execute(query)
        return result.scalar() or 0
    
    async def get_comprehensive_market_analysis(self, db: AsyncSession, skills: List[str],
                                              locations: Optional[List[str]] = None,
                                              time_period_days: int = 90,
                                              include_predictions: bool = True,
                                              include_comparisons: bool = True) -> Dict[str, Any]:
        """
        Get comprehensive market analysis with trends, predictions, and comparisons.
        
        Args:
            db: Database session
            skills: List of skills to analyze
            locations: Optional list of locations to analyze
            time_period_days: Analysis time period
            include_predictions: Include future predictions
            include_comparisons: Include skill comparisons
            
        Returns:
            Comprehensive market analysis data
        """
        try:
            logger.info(f"Generating comprehensive market analysis for {len(skills)} skills")
            
            analysis = {
                'analysis_date': datetime.utcnow().isoformat(),
                'time_period_days': time_period_days,
                'skills_analyzed': skills,
                'locations_analyzed': locations,
                'market_overview': {},
                'skill_trends': [],
                'emerging_skills': [],
                'salary_analysis': [],
                'geographic_analysis': {},
                'skill_comparisons': {}
            }
            
            # Market overview
            analysis['market_overview'] = await self._get_market_overview(db, time_period_days)
            
            # Skill trends analysis
            for skill in skills:
                skill_trend = await self._analyze_skill_trend(db, skill, time_period_days)
                analysis['skill_trends'].append(skill_trend)
            
            # Emerging skills detection
            emerging_skills = await self.detect_emerging_skills(db, time_period_days)
            analysis['emerging_skills'] = [
                {
                    'skill_name': skill.skill_name,
                    'growth_rate': skill.growth_rate,
                    'current_demand': skill.current_demand,
                    'trend_score': skill.trend_score,
                    'confidence': skill.confidence
                }
                for skill in emerging_skills[:10]  # Top 10 emerging skills
            ]
            
            # Salary analysis
            if include_predictions:
                for skill in skills:
                    try:
                        salary_predictions = await self.predict_salaries(
                            db=db,
                            skill_names=[skill],
                            location=locations[0] if locations else None,
                            days=time_period_days
                        )
                        if salary_predictions:
                            analysis['salary_analysis'].append({
                                'skill': skill,
                                'predictions': salary_predictions[0].__dict__
                            })
                    except Exception as e:
                        logger.warning(f"Failed to predict salary for {skill}: {e}")
            
            # Geographic analysis
            if locations:
                analysis['geographic_analysis'] = await self._analyze_geographic_trends(
                    db, skills, locations, time_period_days
                )
            
            # Skill comparisons
            if include_comparisons and len(skills) > 1:
                analysis['skill_comparisons'] = await self._compare_skills(
                    db, skills, time_period_days
                )
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error generating comprehensive market analysis: {e}")
            raise Exception(f"Failed to generate market analysis: {str(e)}")
    
    async def get_skill_market_data(self, db: AsyncSession, skill: str) -> Dict[str, Any]:
        """Get market data for a specific skill."""
        try:
            # Get job postings for this skill
            job_repo = JobRepository()
            
            # Query jobs that mention this skill
            query = select(JobPosting).where(
                JobPosting.processed_skills.has_key(skill),
                JobPosting.is_active == True,
                JobPosting.posted_date >= datetime.utcnow() - timedelta(days=90)
            )
            
            result = await db.execute(query)
            jobs = result.scalars().all()
            
            if not jobs:
                return {
                    'skill': skill,
                    'demand_score': 0.0,
                    'job_count': 0,
                    'avg_salary': None,
                    'growth_trend': 'unknown',
                    'market_competitiveness': 'low'
                }
            
            # Calculate demand score based on job frequency
            total_jobs_query = select(func.count(JobPosting.id)).where(
                JobPosting.is_active == True,
                JobPosting.posted_date >= datetime.utcnow() - timedelta(days=90)
            )
            total_jobs_result = await db.execute(total_jobs_query)
            total_jobs = total_jobs_result.scalar() or 1
            
            demand_score = len(jobs) / total_jobs
            
            # Calculate average salary
            salaries = [job.salary_max for job in jobs if job.salary_max and job.salary_max > 0]
            avg_salary = sum(salaries) / len(salaries) if salaries else None
            
            # Determine growth trend (simplified)
            recent_jobs = [job for job in jobs if job.posted_date >= datetime.utcnow() - timedelta(days=30)]
            older_jobs = [job for job in jobs if job.posted_date < datetime.utcnow() - timedelta(days=30)]
            
            if len(recent_jobs) > len(older_jobs) * 1.2:
                growth_trend = 'growing'
            elif len(recent_jobs) < len(older_jobs) * 0.8:
                growth_trend = 'declining'
            else:
                growth_trend = 'stable'
            
            # Market competitiveness based on demand score
            if demand_score > 0.1:
                competitiveness = 'high'
            elif demand_score > 0.05:
                competitiveness = 'medium'
            else:
                competitiveness = 'low'
            
            return {
                'skill': skill,
                'demand_score': round(demand_score, 4),
                'job_count': len(jobs),
                'avg_salary': round(avg_salary) if avg_salary else None,
                'growth_trend': growth_trend,
                'market_competitiveness': competitiveness,
                'recent_job_count': len(recent_jobs),
                'analysis_date': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting skill market data for {skill}: {e}")
            return {
                'skill': skill,
                'demand_score': 0.0,
                'job_count': 0,
                'error': str(e)
            }
    
    async def get_role_market_data(self, db: AsyncSession, role: str) -> Dict[str, Any]:
        """Get market data for a specific role."""
        try:
            # Query jobs for this role
            query = select(JobPosting).where(
                JobPosting.title.ilike(f"%{role}%"),
                JobPosting.is_active == True,
                JobPosting.posted_date >= datetime.utcnow() - timedelta(days=90)
            )
            
            result = await db.execute(query)
            jobs = result.scalars().all()
            
            if not jobs:
                return {
                    'role': role,
                    'job_count': 0,
                    'avg_salary': None,
                    'demand_level': 'low',
                    'top_skills': [],
                    'locations': []
                }
            
            # Calculate average salary
            salaries = [job.salary_max for job in jobs if job.salary_max and job.salary_max > 0]
            avg_salary = sum(salaries) / len(salaries) if salaries else None
            
            # Get top skills for this role
            skill_counts = {}
            for job in jobs:
                if job.processed_skills:
                    for skill, importance in job.processed_skills.items():
                        skill_counts[skill] = skill_counts.get(skill, 0) + importance
            
            top_skills = sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # Get top locations
            location_counts = {}
            for job in jobs:
                if job.location:
                    location_counts[job.location] = location_counts.get(job.location, 0) + 1
            
            top_locations = sorted(location_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            
            # Determine demand level
            if len(jobs) > 100:
                demand_level = 'high'
            elif len(jobs) > 50:
                demand_level = 'medium'
            else:
                demand_level = 'low'
            
            return {
                'role': role,
                'job_count': len(jobs),
                'avg_salary': round(avg_salary) if avg_salary else None,
                'salary_range': {
                    'min': min(salaries) if salaries else None,
                    'max': max(salaries) if salaries else None
                },
                'demand_level': demand_level,
                'top_skills': [{'skill': skill, 'importance': round(importance, 2)} for skill, importance in top_skills],
                'top_locations': [{'location': loc, 'job_count': count} for loc, count in top_locations],
                'analysis_date': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting role market data for {role}: {e}")
            return {
                'role': role,
                'job_count': 0,
                'error': str(e)
            }
    
    async def _get_market_overview(self, db: AsyncSession, time_period_days: int) -> Dict[str, Any]:
        """Get general market overview statistics."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=time_period_days)
            
            # Total job postings
            total_jobs_query = select(func.count(JobPosting.id)).where(
                JobPosting.posted_date >= cutoff_date,
                JobPosting.is_active == True
            )
            total_jobs_result = await db.execute(total_jobs_query)
            total_jobs = total_jobs_result.scalar() or 0
            
            # Average salary
            avg_salary_query = select(func.avg(JobPosting.salary_max)).where(
                JobPosting.posted_date >= cutoff_date,
                JobPosting.salary_max.isnot(None),
                JobPosting.salary_max > 0
            )
            avg_salary_result = await db.execute(avg_salary_query)
            avg_salary = avg_salary_result.scalar()
            
            # Top companies by job postings
            top_companies_query = select(
                JobPosting.company,
                func.count(JobPosting.id).label('job_count')
            ).where(
                JobPosting.posted_date >= cutoff_date,
                JobPosting.is_active == True
            ).group_by(JobPosting.company).order_by(func.count(JobPosting.id).desc()).limit(10)
            
            top_companies_result = await db.execute(top_companies_query)
            top_companies = [
                {'company': row.company, 'job_count': row.job_count}
                for row in top_companies_result.fetchall()
            ]
            
            return {
                'total_job_postings': total_jobs,
                'average_salary': round(avg_salary) if avg_salary else None,
                'top_hiring_companies': top_companies,
                'analysis_period_days': time_period_days,
                'market_activity_level': 'high' if total_jobs > 1000 else 'medium' if total_jobs > 500 else 'low'
            }
            
        except Exception as e:
            logger.error(f"Error getting market overview: {e}")
            return {
                'total_job_postings': 0,
                'error': str(e)
            }
    
    async def _analyze_skill_trend(self, db: AsyncSession, skill: str, time_period_days: int) -> Dict[str, Any]:
        """Analyze trend for a specific skill."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=time_period_days)
            
            # Get job postings mentioning this skill over time
            query = select(JobPosting).where(
                JobPosting.processed_skills.has_key(skill),
                JobPosting.posted_date >= cutoff_date,
                JobPosting.is_active == True
            ).order_by(JobPosting.posted_date)
            
            result = await db.execute(query)
            jobs = result.scalars().all()
            
            if not jobs:
                return {
                    'skill': skill,
                    'trend': 'no_data',
                    'job_count': 0,
                    'growth_rate': 0.0
                }
            
            # Group by week to analyze trend
            weekly_counts = {}
            for job in jobs:
                week_start = job.posted_date.replace(hour=0, minute=0, second=0, microsecond=0)
                week_start = week_start - timedelta(days=week_start.weekday())
                week_key = week_start.strftime('%Y-%W')
                weekly_counts[week_key] = weekly_counts.get(week_key, 0) + 1
            
            # Calculate trend
            weeks = sorted(weekly_counts.keys())
            if len(weeks) < 2:
                trend = 'stable'
                growth_rate = 0.0
            else:
                counts = [weekly_counts[week] for week in weeks]
                # Simple linear regression to determine trend
                x = np.arange(len(counts))
                slope, _, _, _, _ = stats.linregress(x, counts)
                
                if slope > 0.1:
                    trend = 'growing'
                elif slope < -0.1:
                    trend = 'declining'
                else:
                    trend = 'stable'
                
                growth_rate = slope
            
            return {
                'skill': skill,
                'trend': trend,
                'job_count': len(jobs),
                'growth_rate': round(growth_rate, 3),
                'weekly_data': weekly_counts
            }
            
        except Exception as e:
            logger.error(f"Error analyzing skill trend for {skill}: {e}")
            return {
                'skill': skill,
                'trend': 'error',
                'error': str(e)
            }
    
    async def _analyze_geographic_trends(self, db: AsyncSession, skills: List[str],
                                       locations: List[str], time_period_days: int) -> Dict[str, Any]:
        """Analyze geographic trends for skills and locations."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=time_period_days)
            geographic_data = {}
            
            for location in locations:
                location_data = {
                    'location': location,
                    'total_jobs': 0,
                    'skill_demand': {},
                    'avg_salary': None
                }
                
                # Get jobs in this location
                query = select(JobPosting).where(
                    JobPosting.location.ilike(f"%{location}%"),
                    JobPosting.posted_date >= cutoff_date,
                    JobPosting.is_active == True
                )
                
                result = await db.execute(query)
                location_jobs = result.scalars().all()
                
                location_data['total_jobs'] = len(location_jobs)
                
                # Calculate average salary for location
                salaries = [job.salary_max for job in location_jobs if job.salary_max and job.salary_max > 0]
                if salaries:
                    location_data['avg_salary'] = round(sum(salaries) / len(salaries))
                
                # Analyze skill demand in this location
                for skill in skills:
                    skill_jobs = [
                        job for job in location_jobs
                        if job.processed_skills and skill in job.processed_skills
                    ]
                    location_data['skill_demand'][skill] = {
                        'job_count': len(skill_jobs),
                        'demand_ratio': len(skill_jobs) / len(location_jobs) if location_jobs else 0
                    }
                
                geographic_data[location] = location_data
            
            return geographic_data
            
        except Exception as e:
            logger.error(f"Error analyzing geographic trends: {e}")
            return {}
    
    async def _compare_skills(self, db: AsyncSession, skills: List[str], time_period_days: int) -> Dict[str, Any]:
        """Compare multiple skills across various metrics."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=time_period_days)
            comparison_data = {
                'skills_compared': skills,
                'metrics': {
                    'job_demand': {},
                    'salary_potential': {},
                    'growth_trend': {},
                    'market_share': {}
                }
            }
            
            total_jobs_query = select(func.count(JobPosting.id)).where(
                JobPosting.posted_date >= cutoff_date,
                JobPosting.is_active == True
            )
            total_jobs_result = await db.execute(total_jobs_query)
            total_jobs = total_jobs_result.scalar() or 1
            
            for skill in skills:
                # Job demand
                skill_jobs_query = select(func.count(JobPosting.id)).where(
                    JobPosting.processed_skills.has_key(skill),
                    JobPosting.posted_date >= cutoff_date,
                    JobPosting.is_active == True
                )
                skill_jobs_result = await db.execute(skill_jobs_query)
                skill_job_count = skill_jobs_result.scalar() or 0
                
                comparison_data['metrics']['job_demand'][skill] = skill_job_count
                comparison_data['metrics']['market_share'][skill] = round(skill_job_count / total_jobs, 4)
                
                # Salary potential
                salary_query = select(func.avg(JobPosting.salary_max)).where(
                    JobPosting.processed_skills.has_key(skill),
                    JobPosting.posted_date >= cutoff_date,
                    JobPosting.salary_max.isnot(None),
                    JobPosting.salary_max > 0
                )
                salary_result = await db.execute(salary_query)
                avg_salary = salary_result.scalar()
                
                comparison_data['metrics']['salary_potential'][skill] = round(avg_salary) if avg_salary else None
                
                # Growth trend (simplified)
                recent_jobs_query = select(func.count(JobPosting.id)).where(
                    JobPosting.processed_skills.has_key(skill),
                    JobPosting.posted_date >= datetime.utcnow() - timedelta(days=30),
                    JobPosting.is_active == True
                )
                recent_jobs_result = await db.execute(recent_jobs_query)
                recent_count = recent_jobs_result.scalar() or 0
                
                older_jobs_query = select(func.count(JobPosting.id)).where(
                    JobPosting.processed_skills.has_key(skill),
                    JobPosting.posted_date >= cutoff_date,
                    JobPosting.posted_date < datetime.utcnow() - timedelta(days=30),
                    JobPosting.is_active == True
                )
                older_jobs_result = await db.execute(older_jobs_query)
                older_count = older_jobs_result.scalar() or 0
                
                if older_count > 0:
                    growth_rate = (recent_count - older_count) / older_count
                else:
                    growth_rate = 0.0
                
                comparison_data['metrics']['growth_trend'][skill] = round(growth_rate, 3)
            
            # Add rankings
            comparison_data['rankings'] = {
                'most_in_demand': sorted(
                    comparison_data['metrics']['job_demand'].items(),
                    key=lambda x: x[1], reverse=True
                ),
                'highest_salary': sorted(
                    [(k, v) for k, v in comparison_data['metrics']['salary_potential'].items() if v],
                    key=lambda x: x[1], reverse=True
                ),
                'fastest_growing': sorted(
                    comparison_data['metrics']['growth_trend'].items(),
                    key=lambda x: x[1], reverse=True
                )
            }
            
            return comparison_data
            
        except Exception as e:
            logger.error(f"Error comparing skills: {e}")
            return {}