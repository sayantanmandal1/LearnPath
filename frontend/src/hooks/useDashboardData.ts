/**
 * Custom hook for managing enhanced dashboard data with real-time integration
 */
import { useState, useEffect, useCallback } from 'react';

// Import API service
import { apiService } from '../services/api';

// Mock data functions for when user is not authenticated or analysis not completed
const getMockDashboardData = () => ({
    user_profile: {
        name: 'Demo User',
        current_role: 'Software Developer',
        experience_years: 0,
        skills: []
    },
    analysis_status: 'pending',
    message: 'Complete your profile analysis to unlock personalized insights'
});

const getMockSkillRadarData = () => ({
    skills: [
        { name: 'JavaScript', current_level: 0, target_level: 0, market_average: 0 },
        { name: 'React', current_level: 0, target_level: 0, market_average: 0 },
        { name: 'Python', current_level: 0, target_level: 0, market_average: 0 },
        { name: 'Node.js', current_level: 0, target_level: 0, market_average: 0 },
        { name: 'SQL', current_level: 0, target_level: 0, market_average: 0 },
        { name: 'AWS', current_level: 0, target_level: 0, market_average: 0 }
    ]
});

const getMockCareerProgressData = () => ({
    career_score_trend: [
        { date: new Date().toISOString(), score: 0 }
    ],
    skill_improvements: []
});

const getMockJobRecommendations = () => ({
    job_matches: []
});

const getMockMarketInsights = () => ({
    trends: [],
    salary_insights: [],
    skill_demands: []
});

const getMockPersonalizedContent = () => ({
    suggested_learning_paths: []
});

// TypeScript interfaces for the hook
interface DashboardOptions {
    autoRefresh?: boolean;
    refreshInterval?: number;
    includeJobMatches?: boolean;
    includeSkillRadar?: boolean;
    includeCareerProgress?: boolean;
    includeMarketInsights?: boolean;
    preferredCities?: string[];
    targetRole?: string | null;
}

interface SkillData {
    name: string;
    current_level?: number;
    target_level?: number;
    market_average?: number;
}

interface SkillRadarData {
    skills?: SkillData[];
}

interface CareerScoreTrend {
    date: string;
    score: number;
}

interface CareerProgressData {
    career_score_trend?: CareerScoreTrend[];
    skill_improvements?: Array<{
        skill: string;
        category: string;
        improvement: number;
    }>;
}

interface JobMatch {
    match_score: number;
    skill_gaps?: any[];
    gap_analysis?: any;
    [key: string]: any;
}

interface JobRecommendations {
    job_matches?: JobMatch[];
}

interface MarketTrend {
    relevance_score: number;
    impact?: string;
    [key: string]: any;
}

interface MarketInsights {
    trends?: MarketTrend[];
    salary_insights?: any[];
    skill_demands?: any[];
}

interface LearningPath {
    progress?: number;
    estimated_duration?: string;
    [key: string]: any;
}

interface PersonalizedContent {
    suggested_learning_paths?: LearningPath[];
}

interface ProcessedSkillData {
    skill: string;
    current: number;
    target: number;
    market_average: number;
    fullMark: number;
}

interface ProcessedCareerData {
    date: string;
    score: number;
    timestamp: string;
}

interface ProcessedJobMatch extends JobMatch {
    match_percentage: number;
}

interface ProcessedMarketTrend extends MarketTrend {
    relevance_percentage: number;
    impact_level: string;
}

interface ProcessedLearningPath extends LearningPath {
    progress_percentage: number;
    estimated_weeks: number;
}

export const useDashboardData = (options: DashboardOptions = {}) => {
    const [dashboardData, setDashboardData] = useState<any>(null);
    const [skillRadarData, setSkillRadarData] = useState<SkillRadarData | null>(null);
    const [careerProgressData, setCareerProgressData] = useState<CareerProgressData | null>(null);
    const [jobRecommendations, setJobRecommendations] = useState<JobRecommendations | null>(null);
    const [marketInsights, setMarketInsights] = useState<MarketInsights | null>(null);
    const [personalizedContent, setPersonalizedContent] = useState<PersonalizedContent | null>(null);
    const [loading, setLoading] = useState<boolean>(true);
    const [error, setError] = useState<string | null>(null);
    const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

    const {
        autoRefresh = false,
        refreshInterval = 300000, // 5 minutes
        includeJobMatches = true,
        includeSkillRadar = true,
        includeCareerProgress = true,
        includeMarketInsights = true,
        preferredCities = ['Bangalore', 'Hyderabad', 'Pune', 'Chennai', 'Mumbai', 'Delhi NCR'],
        targetRole = null,
    } = options;

    // Load comprehensive dashboard data
    const loadDashboardData = useCallback(async (forceRefresh = false) => {
        try {
            setLoading(true);
            setError(null);

            // Check if user is authenticated first
            const token = localStorage.getItem('access_token');
            if (!token) {
                // User not authenticated, show mock data
                setDashboardData(getMockDashboardData());
                setSkillRadarData(getMockSkillRadarData());
                setCareerProgressData(getMockCareerProgressData());
                setJobRecommendations(getMockJobRecommendations());
                setMarketInsights(getMockMarketInsights());
                setPersonalizedContent(getMockPersonalizedContent());
                setLastUpdated(new Date());
                setLoading(false);
                return;
            }

            // Get comprehensive dashboard data
            const comprehensiveData = await apiService.getComprehensiveDashboardData({
                includeJobMatches,
                includeSkillRadar,
                includeCareerProgress,
            });

            setDashboardData(comprehensiveData);

            // Load additional data components
            const dataPromises: Promise<void>[] = [];

            // Load skill radar data if requested
            if (includeSkillRadar) {
                dataPromises.push(
                    apiService.getSkillRadarData({
                        includeMarketComparison: true,
                    }).then(data => setSkillRadarData(data))
                );
            }

            // Load career progress data if requested
            if (includeCareerProgress) {
                dataPromises.push(
                    apiService.getCareerProgressTracking({
                        trackingPeriodDays: 90,
                        includePredictions: true,
                        includeMilestones: true,
                    }).then(data => setCareerProgressData(data))
                );
            }

            // Load job recommendations if requested
            if (includeJobMatches) {
                dataPromises.push(
                    apiService.getPersonalizedJobRecommendations({
                        limit: 20,
                        preferredCities,
                        targetRole,
                        minMatchScore: 0.6,
                        includeMarketInsights: true,
                    }).then(data => setJobRecommendations(data))
                );
            }

            // Load market insights if requested
            if (includeMarketInsights) {
                dataPromises.push(
                    apiService.getMarketInsights({
                        role: targetRole,
                        location: 'India',
                    }).then(data => setMarketInsights(data))
                );
            }

            // Load personalized content
            dataPromises.push(
                apiService.getPersonalizedContent().then(data => setPersonalizedContent(data))
            );

            // Wait for all additional data to load
            await Promise.allSettled(dataPromises);

            setLastUpdated(new Date());
        } catch (err: any) {
            console.error('Error loading dashboard data:', err);
            
            // If authentication error, show mock data instead of error
            if (err.message?.includes('Not authenticated') || err.message?.includes('403')) {
                console.log('Authentication failed, showing mock data');
                setDashboardData(getMockDashboardData());
                setSkillRadarData(getMockSkillRadarData());
                setCareerProgressData(getMockCareerProgressData());
                setJobRecommendations(getMockJobRecommendations());
                setMarketInsights(getMockMarketInsights());
                setPersonalizedContent(getMockPersonalizedContent());
                setLastUpdated(new Date());
                setError(null); // Clear error to show mock data
            } else {
                setError(err.message || 'Failed to load dashboard data');
            }
        } finally {
            setLoading(false);
        }
    }, [includeJobMatches, includeSkillRadar, includeCareerProgress, includeMarketInsights, preferredCities, targetRole]);

    // Refresh specific data components
    const refreshSkillRadar = useCallback(async () => {
        try {
            const token = localStorage.getItem('access_token');
            if (!token) {
                setSkillRadarData(getMockSkillRadarData());
                return;
            }
            
            const data = await apiService.getSkillRadarData({
                includeMarketComparison: true,
            });
            setSkillRadarData(data);
        } catch (err) {
            console.error('Error refreshing skill radar data:', err);
            if (err.message?.includes('Not authenticated')) {
                setSkillRadarData(getMockSkillRadarData());
            }
        }
    }, []);

    const refreshJobRecommendations = useCallback(async () => {
        try {
            const token = localStorage.getItem('access_token');
            if (!token) {
                setJobRecommendations(getMockJobRecommendations());
                return;
            }
            
            const data = await apiService.getPersonalizedJobRecommendations({
                limit: 20,
                preferredCities,
                targetRole,
                minMatchScore: 0.6,
                includeMarketInsights: true,
            });
            setJobRecommendations(data);
        } catch (err) {
            console.error('Error refreshing job recommendations:', err);
            if (err.message?.includes('Not authenticated')) {
                setJobRecommendations(getMockJobRecommendations());
            }
        }
    }, [preferredCities, targetRole]);

    const refreshCareerProgress = useCallback(async () => {
        try {
            const token = localStorage.getItem('access_token');
            if (!token) {
                setCareerProgressData(getMockCareerProgressData());
                return;
            }
            
            const data = await apiService.getCareerProgressTracking({
                trackingPeriodDays: 90,
                includePredictions: true,
                includeMilestones: true,
            });
            setCareerProgressData(data);
        } catch (err) {
            console.error('Error refreshing career progress data:', err);
            if (err.message?.includes('Not authenticated')) {
                setCareerProgressData(getMockCareerProgressData());
            }
        }
    }, []);

    const refreshMarketInsights = useCallback(async () => {
        try {
            const token = localStorage.getItem('access_token');
            if (!token) {
                setMarketInsights(getMockMarketInsights());
                return;
            }
            
            const data = await apiService.getMarketInsights({
                role: targetRole,
                location: 'India',
            });
            setMarketInsights(data);
        } catch (err) {
            console.error('Error refreshing market insights:', err);
            if (err.message?.includes('Not authenticated')) {
                setMarketInsights(getMockMarketInsights());
            }
        }
    }, [targetRole]);

    // Get real-time analysis results
    const getAnalysisResults = useCallback(async (forceRefresh = false) => {
        try {
            const data = await apiService.getRealTimeAnalysisResults({
                forceRefresh,
                analysisTypes: ['skill_assessment', 'career_recommendations', 'learning_paths'],
            });
            return data;
        } catch (err) {
            console.error('Error getting analysis results:', err);
            throw err;
        }
    }, []);

    // Initial data load
    useEffect(() => {
        loadDashboardData();
    }, [loadDashboardData]);

    // Auto-refresh setup - only if authenticated
    useEffect(() => {
        if (!autoRefresh) return;
        
        // Don't auto-refresh if not authenticated
        const token = localStorage.getItem('access_token');
        if (!token) return;

        const interval = setInterval(() => {
            // Check token again before each refresh
            const currentToken = localStorage.getItem('access_token');
            if (currentToken) {
                loadDashboardData();
            }
        }, refreshInterval);

        return () => clearInterval(interval);
    }, [autoRefresh, refreshInterval, loadDashboardData]);

    // Helper functions for data processing
    const getSkillRadarChartData = useCallback((): ProcessedSkillData[] => {
        if (!skillRadarData?.skills) return [];

        return skillRadarData.skills.map(skill => {
            const currentLevel = skill.current_level || 0;
            return {
                skill: skill.name,
                current: currentLevel,
                target: skill.target_level || currentLevel + 20,
                market_average: skill.market_average || 50,
                fullMark: 100,
            };
        });
    }, [skillRadarData]);

    const getCareerProgressChartData = useCallback((): ProcessedCareerData[] => {
        if (!careerProgressData?.career_score_trend) return [];

        return careerProgressData.career_score_trend.map(point => ({
            date: new Date(point.date).toLocaleDateString(),
            score: point.score,
            timestamp: point.date,
        }));
    }, [careerProgressData]);

    const getJobMatchesWithGaps = useCallback((): ProcessedJobMatch[] => {
        if (!jobRecommendations?.job_matches) return [];

        return jobRecommendations.job_matches.map(job => ({
            ...job,
            skill_gaps: job.skill_gaps || [],
            match_percentage: Math.round(job.match_score * 100),
            gap_analysis: job.gap_analysis || {},
        }));
    }, [jobRecommendations]);

    const getMarketTrends = useCallback((): ProcessedMarketTrend[] => {
        if (!marketInsights?.trends) return [];

        return marketInsights.trends.map(trend => ({
            ...trend,
            relevance_percentage: Math.round(trend.relevance_score * 10),
            impact_level: trend.impact || 'Medium',
        }));
    }, [marketInsights]);

    const getLearningPathsWithProgress = useCallback((): ProcessedLearningPath[] => {
        if (!personalizedContent?.suggested_learning_paths) return [];

        return personalizedContent.suggested_learning_paths.map(path => ({
            ...path,
            progress_percentage: path.progress || 0,
            estimated_weeks: path.estimated_duration ?
                parseInt(path.estimated_duration.replace(/\D/g, '')) || 4 : 4,
        }));
    }, [personalizedContent]);

    return {
        // Data states
        dashboardData,
        skillRadarData,
        careerProgressData,
        jobRecommendations,
        marketInsights,
        personalizedContent,

        // Loading and error states
        loading,
        error,
        lastUpdated,

        // Actions
        loadDashboardData,
        refreshSkillRadar,
        refreshJobRecommendations,
        refreshCareerProgress,
        refreshMarketInsights,
        getAnalysisResults,

        // Processed data helpers
        getSkillRadarChartData,
        getCareerProgressChartData,
        getJobMatchesWithGaps,
        getMarketTrends,
        getLearningPathsWithProgress,

        // Computed values
        isDataFresh: lastUpdated && (Date.now() - lastUpdated.getTime()) < refreshInterval,
        hasJobRecommendations: (jobRecommendations?.job_matches?.length ?? 0) > 0,
        hasSkillData: (skillRadarData?.skills?.length ?? 0) > 0,
        hasCareerProgress: (careerProgressData?.career_score_trend?.length ?? 0) > 0,
        hasMarketInsights: (marketInsights?.trends?.length ?? 0) > 0,
    };
};

export default useDashboardData;