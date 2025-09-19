/**
 * Job Recommendations Card with Indian market focus and real-time data
 */
import React, { useState } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Badge } from '../ui/badge';
import { Button } from '../ui/button';
import { Progress } from '../ui/progress';
import { 
  MapPin, 
  Building, 
  DollarSign, 
  TrendingUp, 
  Clock, 
  ExternalLink,
  RefreshCw,
  AlertCircle,
  CheckCircle,
  Target,
  Users
} from 'lucide-react';

interface JobMatch {
  id: string;
  title: string;
  company: string;
  location: string;
  salary_range?: string;
  match_score: number;
  skill_matches: string[];
  skill_gaps: string[];
  posted_date: string;
  job_type: string;
  experience_level: string;
  url?: string;
  gap_analysis?: {
    missing_skills: string[];
    skill_strength: number;
    experience_gap: number;
  };
}

interface JobRecommendationsCardProps {
  jobMatches: JobMatch[];
  marketInsights?: {
    salary_trends: any;
    demand_trends: any;
    top_skills: string[];
  };
  loading?: boolean;
  onRefresh?: () => void;
  onJobClick?: (job: JobMatch) => void;
  className?: string;
}

export const JobRecommendationsCard: React.FC<JobRecommendationsCardProps> = ({
  jobMatches = [],
  marketInsights,
  loading = false,
  onRefresh,
  onJobClick,
  className = '',
}) => {
  const [selectedJob, setSelectedJob] = useState<JobMatch | null>(null);
  const [showGapAnalysis, setShowGapAnalysis] = useState(false);

  const handleJobSelect = (job: JobMatch) => {
    setSelectedJob(job);
    setShowGapAnalysis(true);
    onJobClick?.(job);
  };

  const getMatchColor = (score: number) => {
    if (score >= 0.8) return 'text-green-600 bg-green-50 border-green-200';
    if (score >= 0.6) return 'text-blue-600 bg-blue-50 border-blue-200';
    return 'text-orange-600 bg-orange-50 border-orange-200';
  };

  const getExperienceLevel = (level: string) => {
    const levels: { [key: string]: string } = {
      'entry': 'Entry Level',
      'mid': 'Mid Level',
      'senior': 'Senior Level',
      'lead': 'Lead/Principal',
    };
    return levels[level.toLowerCase()] || level;
  };

  const formatSalary = (salaryRange: string) => {
    if (!salaryRange) return 'Not disclosed';
    return salaryRange.replace(/(\d+)k/g, '₹$1,000').replace(/(\d+)L/g, '₹$1 Lakh');
  };

  const getLocationIcon = (location: string) => {
    const indianCities = ['bangalore', 'hyderabad', 'pune', 'chennai', 'mumbai', 'delhi', 'gurgaon', 'noida'];
    const isIndianCity = indianCities.some(city => location.toLowerCase().includes(city));
    return isIndianCity ? '🇮🇳' : '🌍';
  };

  return (
    <Card className={`h-full ${className}`}>
      <CardHeader className="pb-4">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center space-x-2">
            <Target className="w-5 h-5 text-blue-600" />
            <span>Job Recommendations</span>
            <Badge variant="secondary" className="ml-2">
              {jobMatches.length} matches
            </Badge>
          </CardTitle>
          <Button
            variant="ghost"
            size="sm"
            onClick={onRefresh}
            disabled={loading}
            className="flex items-center space-x-1"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
            <span>Refresh</span>
          </Button>
        </div>
        
        {marketInsights && (
          <div className="flex items-center space-x-4 text-sm text-muted-foreground mt-2">
            <div className="flex items-center space-x-1">
              <TrendingUp className="w-4 h-4" />
              <span>Market trending up</span>
            </div>
            <div className="flex items-center space-x-1">
              <Users className="w-4 h-4" />
              <span>High demand in India</span>
            </div>
          </div>
        )}
      </CardHeader>

      <CardContent className="space-y-4">
        {loading ? (
          <div className="space-y-4">
            {[1, 2, 3].map((i) => (
              <div key={i} className="animate-pulse">
                <div className="h-4 bg-gray-200 rounded w-3/4 mb-2"></div>
                <div className="h-3 bg-gray-200 rounded w-1/2 mb-2"></div>
                <div className="h-2 bg-gray-200 rounded w-full"></div>
              </div>
            ))}
          </div>
        ) : jobMatches.length === 0 ? (
          <div className="text-center py-8 text-muted-foreground">
            <Target className="w-12 h-12 mx-auto mb-4 opacity-50" />
            <p>No job recommendations available</p>
            <p className="text-sm">Complete your profile to get personalized matches</p>
          </div>
        ) : (
          <div className="space-y-4 max-h-96 overflow-y-auto">
            <AnimatePresence>
              {jobMatches.map((job, index) => (
                <motion.div
                  key={job.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  transition={{ duration: 0.3, delay: index * 0.1 }}
                  className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow cursor-pointer"
                  onClick={() => handleJobSelect(job)}
                >
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex-1">
                      <h4 className="font-semibold text-gray-900 mb-1">{job.title}</h4>
                      <div className="flex items-center space-x-2 text-sm text-muted-foreground mb-2">
                        <Building className="w-4 h-4" />
                        <span>{job.company}</span>
                        <span>•</span>
                        <MapPin className="w-4 h-4" />
                        <span>{getLocationIcon(job.location)} {job.location}</span>
                      </div>
                    </div>
                    <Badge className={`${getMatchColor(job.match_score)} border`}>
                      {Math.round(job.match_score * 100)}% match
                    </Badge>
                  </div>

                  <div className="space-y-2 mb-3">
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-muted-foreground">Match Score</span>
                      <span className="font-medium">{Math.round(job.match_score * 100)}%</span>
                    </div>
                    <Progress value={job.match_score * 100} className="h-2" />
                  </div>

                  <div className="flex items-center justify-between text-sm">
                    <div className="flex items-center space-x-4">
                      {job.salary_range && (
                        <div className="flex items-center space-x-1 text-green-600">
                          <DollarSign className="w-4 h-4" />
                          <span>{formatSalary(job.salary_range)}</span>
                        </div>
                      )}
                      <div className="flex items-center space-x-1 text-muted-foreground">
                        <Clock className="w-4 h-4" />
                        <span>{getExperienceLevel(job.experience_level)}</span>
                      </div>
                    </div>
                    {job.url && (
                      <Button variant="ghost" size="sm" className="p-1">
                        <ExternalLink className="w-4 h-4" />
                      </Button>
                    )}
                  </div>

                  {/* Skill matches and gaps preview */}
                  <div className="mt-3 pt-3 border-t border-gray-100">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-2">
                        <CheckCircle className="w-4 h-4 text-green-600" />
                        <span className="text-sm text-muted-foreground">
                          {job.skill_matches?.length || 0} skills match
                        </span>
                      </div>
                      {job.skill_gaps?.length > 0 && (
                        <div className="flex items-center space-x-2">
                          <AlertCircle className="w-4 h-4 text-orange-600" />
                          <span className="text-sm text-muted-foreground">
                            {job.skill_gaps.length} gaps to fill
                          </span>
                        </div>
                      )}
                    </div>
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        )}

        {/* Gap Analysis Modal/Expanded View */}
        <AnimatePresence>
          {showGapAnalysis && selectedJob && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="border-t border-gray-200 pt-4 mt-4"
            >
              <div className="flex items-center justify-between mb-3">
                <h5 className="font-semibold">Gap Analysis: {selectedJob.title}</h5>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setShowGapAnalysis(false)}
                >
                  ×
                </Button>
              </div>

              <div className="space-y-3">
                {/* Skill Matches */}
                {selectedJob.skill_matches?.length > 0 && (
                  <div>
                    <h6 className="text-sm font-medium text-green-700 mb-2 flex items-center">
                      <CheckCircle className="w-4 h-4 mr-1" />
                      Matching Skills
                    </h6>
                    <div className="flex flex-wrap gap-1">
                      {selectedJob.skill_matches.map((skill, idx) => (
                        <Badge key={idx} variant="secondary" className="text-xs bg-green-50 text-green-700">
                          {skill}
                        </Badge>
                      ))}
                    </div>
                  </div>
                )}

                {/* Skill Gaps */}
                {selectedJob.skill_gaps?.length > 0 && (
                  <div>
                    <h6 className="text-sm font-medium text-orange-700 mb-2 flex items-center">
                      <AlertCircle className="w-4 h-4 mr-1" />
                      Skills to Develop
                    </h6>
                    <div className="flex flex-wrap gap-1">
                      {selectedJob.skill_gaps.map((skill, idx) => (
                        <Badge key={idx} variant="outline" className="text-xs border-orange-200 text-orange-700">
                          {skill}
                        </Badge>
                      ))}
                    </div>
                  </div>
                )}

                {/* Gap Analysis Details */}
                {selectedJob.gap_analysis && (
                  <div className="bg-gray-50 rounded-lg p-3">
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <span className="text-muted-foreground">Skill Strength:</span>
                        <div className="flex items-center space-x-2 mt-1">
                          <Progress 
                            value={selectedJob.gap_analysis.skill_strength * 100} 
                            className="h-2 flex-1" 
                          />
                          <span className="font-medium">
                            {Math.round(selectedJob.gap_analysis.skill_strength * 100)}%
                          </span>
                        </div>
                      </div>
                      <div>
                        <span className="text-muted-foreground">Experience Match:</span>
                        <div className="flex items-center space-x-2 mt-1">
                          <Progress 
                            value={(1 - selectedJob.gap_analysis.experience_gap) * 100} 
                            className="h-2 flex-1" 
                          />
                          <span className="font-medium">
                            {Math.round((1 - selectedJob.gap_analysis.experience_gap) * 100)}%
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Market Insights Summary */}
        {marketInsights && (
          <div className="border-t border-gray-200 pt-4 mt-4">
            <h5 className="font-semibold mb-2 flex items-center">
              <TrendingUp className="w-4 h-4 mr-2" />
              Market Insights
            </h5>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm">
              {marketInsights.top_skills?.length > 0 && (
                <div>
                  <span className="text-muted-foreground">In-Demand Skills:</span>
                  <div className="flex flex-wrap gap-1 mt-1">
                    {marketInsights.top_skills.slice(0, 3).map((skill, idx) => (
                      <Badge key={idx} variant="secondary" className="text-xs">
                        {skill}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}
              <div>
                <span className="text-muted-foreground">Market Status:</span>
                <div className="flex items-center space-x-1 mt-1">
                  <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                  <span className="text-green-600 font-medium">Growing</span>
                </div>
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default JobRecommendationsCard;