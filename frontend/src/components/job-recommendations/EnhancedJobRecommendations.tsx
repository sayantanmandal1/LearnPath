/**
 * Enhanced Job Recommendations with Application Tracking
 */
import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Badge } from '../ui/badge';
import { Button } from '../ui/button';
import { Progress } from '../ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../ui/tabs';
import { 
  MapPin, 
  Building, 
  DollarSign, 
  Clock, 
  ExternalLink,
  RefreshCw,
  AlertCircle,
  CheckCircle,
  Target,
  BookmarkPlus,
  Send,
  Eye,
  Filter,
  BarChart3,
  Briefcase
} from 'lucide-react';

import { jobRecommendationService, EnhancedJobMatch, JobApplicationStats } from '../../services/jobRecommendationService';
import JobApplicationTracker from './JobApplicationTracker';
import JobMarketInsights from './JobMarketInsights';

interface EnhancedJobRecommendationsProps {
  targetRole?: string;
  preferredCities?: string[];
  className?: string;
}

export const EnhancedJobRecommendations: React.FC<EnhancedJobRecommendationsProps> = ({
  targetRole = 'Software Developer',
  preferredCities = ['Bangalore', 'Hyderabad', 'Pune'],
  className = '',
}) => {
  const [jobMatches, setJobMatches] = useState<EnhancedJobMatch[]>([]);
  const [applicationStats, setApplicationStats] = useState<JobApplicationStats | null>(null);
  const [loading, setLoading] = useState(false);
  const [selectedJob, setSelectedJob] = useState<EnhancedJobMatch | null>(null);
  const [showGapAnalysis, setShowGapAnalysis] = useState(false);
  const [activeTab, setActiveTab] = useState('recommendations');
  const [filters, setFilters] = useState({
    minMatchScore: 0.6,
    maxCompetition: 'high',
    indianCitiesOnly: true,
    remoteAcceptable: false
  });

  useEffect(() => {
    loadJobRecommendations();
    loadApplicationStats();
  }, [targetRole, preferredCities]);

  const loadJobRecommendations = async () => {
    setLoading(true);
    try {
      const recommendations = await jobRecommendationService.getLocationBasedJobs(
        targetRole,
        preferredCities,
        {
          remoteAcceptable: filters.remoteAcceptable,
          hybridAcceptable: true,
          limit: 50
        }
      );
      
      // Apply filters
      const filteredRecommendations = recommendations.filter(job => {
        if (job.match_score < filters.minMatchScore) return false;
        if (filters.indianCitiesOnly && !job.is_indian_tech_city) return false;
        return true;
      });
      
      setJobMatches(filteredRecommendations);
    } catch (error) {
      console.error('Error loading job recommendations:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadApplicationStats = async () => {
    try {
      const stats = await jobRecommendationService.getApplicationStats();
      setApplicationStats(stats);
    } catch (error) {
      console.error('Error loading application stats:', error);
    }
  };

  const handleJobInteraction = async (job: EnhancedJobMatch, action: 'viewed' | 'interested' | 'not_interested') => {
    try {
      await jobRecommendationService.trackRecommendationInteraction(job.job_posting_id, action);
      
      if (action === 'interested') {
        // Create application record
        await jobRecommendationService.createApplication({
          job_posting_id: job.job_posting_id,
          job_title: job.job_title,
          company_name: job.company_name,
          job_url: job.job_url,
          match_score: job.match_score,
          skill_matches: job.skill_matches,
          skill_gaps: job.skill_gaps
        });
        
        // Refresh data
        await loadJobRecommendations();
        await loadApplicationStats();
      }
    } catch (error) {
      console.error('Error handling job interaction:', error);
    }
  };

  const handleApplyExternal = async (job: EnhancedJobMatch) => {
    try {
      await jobRecommendationService.markJobAsApplied(job.job_posting_id, 'external');
      await loadJobRecommendations();
      await loadApplicationStats();
      
      // Open job URL
      if (job.job_url) {
        window.open(job.job_url, '_blank');
      }
    } catch (error) {
      console.error('Error marking job as applied:', error);
    }
  };

  const getMatchColor = (score: number) => {
    if (score >= 0.8) return 'text-green-600 bg-green-50 border-green-200';
    if (score >= 0.6) return 'text-blue-600 bg-blue-50 border-blue-200';
    return 'text-orange-600 bg-orange-50 border-orange-200';
  };

  const getApplicationStatusColor = (status?: string) => {
    switch (status) {
      case 'applied': return 'text-blue-600 bg-blue-50';
      case 'interviewing': return 'text-purple-600 bg-purple-50';
      case 'rejected': return 'text-red-600 bg-red-50';
      case 'accepted': return 'text-green-600 bg-green-50';
      case 'interested': return 'text-yellow-600 bg-yellow-50';
      default: return 'text-gray-600 bg-gray-50';
    }
  };

  const formatApplicationStatus = (status?: string) => {
    if (!status) return null;
    return status.charAt(0).toUpperCase() + status.slice(1);
  };

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Header with Stats */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">Job Recommendations</h2>
          <p className="text-gray-600">
            Personalized matches for {targetRole} in {preferredCities.join(', ')}
          </p>
        </div>
        <div className="flex items-center space-x-4">
          {applicationStats && (
            <div className="text-sm text-gray-600">
              <span className="font-medium">{applicationStats.total_applications}</span> applications
              <span className="mx-2">•</span>
              <span className="font-medium">{applicationStats.success_rate}%</span> success rate
            </div>
          )}
          <Button
            variant="outline"
            size="sm"
            onClick={loadJobRecommendations}
            disabled={loading}
            className="flex items-center space-x-2"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
            <span>Refresh</span>
          </Button>
        </div>
      </div>

      {/* Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="recommendations" className="flex items-center space-x-2">
            <Target className="w-4 h-4" />
            <span>Recommendations</span>
          </TabsTrigger>
          <TabsTrigger value="applications" className="flex items-center space-x-2">
            <Briefcase className="w-4 h-4" />
            <span>My Applications</span>
          </TabsTrigger>
          <TabsTrigger value="insights" className="flex items-center space-x-2">
            <BarChart3 className="w-4 h-4" />
            <span>Market Insights</span>
          </TabsTrigger>
        </TabsList>

        {/* Job Recommendations Tab */}
        <TabsContent value="recommendations" className="space-y-4">
          {/* Filters */}
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="flex items-center space-x-2 text-lg">
                <Filter className="w-5 h-5" />
                <span>Filters</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div>
                  <label className="text-sm font-medium text-gray-700">Min Match Score</label>
                  <select 
                    value={filters.minMatchScore}
                    onChange={(e) => setFilters({...filters, minMatchScore: parseFloat(e.target.value)})}
                    className="mt-1 block w-full rounded-md border-gray-300 shadow-sm"
                  >
                    <option value={0.5}>50%+</option>
                    <option value={0.6}>60%+</option>
                    <option value={0.7}>70%+</option>
                    <option value={0.8}>80%+</option>
                  </select>
                </div>
                <div className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    id="indianCities"
                    checked={filters.indianCitiesOnly}
                    onChange={(e) => setFilters({...filters, indianCitiesOnly: e.target.checked})}
                    className="rounded border-gray-300"
                  />
                  <label htmlFor="indianCities" className="text-sm font-medium text-gray-700">
                    Indian Tech Cities Only
                  </label>
                </div>
                <div className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    id="remoteAcceptable"
                    checked={filters.remoteAcceptable}
                    onChange={(e) => setFilters({...filters, remoteAcceptable: e.target.checked})}
                    className="rounded border-gray-300"
                  />
                  <label htmlFor="remoteAcceptable" className="text-sm font-medium text-gray-700">
                    Include Remote Jobs
                  </label>
                </div>
                <Button onClick={loadJobRecommendations} size="sm">
                  Apply Filters
                </Button>
              </div>
            </CardContent>
          </Card>

          {/* Job Recommendations List */}
          <div className="space-y-4">
            {loading ? (
              <div className="space-y-4">
                {[1, 2, 3].map((i) => (
                  <Card key={i} className="animate-pulse">
                    <CardContent className="p-6">
                      <div className="h-4 bg-gray-200 rounded w-3/4 mb-2"></div>
                      <div className="h-3 bg-gray-200 rounded w-1/2 mb-2"></div>
                      <div className="h-2 bg-gray-200 rounded w-full"></div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            ) : jobMatches.length === 0 ? (
              <Card>
                <CardContent className="text-center py-12">
                  <Target className="w-12 h-12 mx-auto mb-4 text-gray-400" />
                  <h3 className="text-lg font-medium text-gray-900 mb-2">No recommendations found</h3>
                  <p className="text-gray-600 mb-4">
                    Try adjusting your filters or target role to see more opportunities.
                  </p>
                  <Button onClick={loadJobRecommendations}>
                    Refresh Recommendations
                  </Button>
                </CardContent>
              </Card>
            ) : (
              <div className="space-y-4">
                <AnimatePresence>
                  {jobMatches.map((job, index) => (
                    <motion.div
                      key={job.job_posting_id}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -20 }}
                      transition={{ duration: 0.3, delay: index * 0.1 }}
                    >
                      <Card className="hover:shadow-lg transition-shadow cursor-pointer">
                        <CardContent className="p-6">
                          <div className="flex items-start justify-between mb-4">
                            <div className="flex-1">
                              <div className="flex items-center space-x-2 mb-2">
                                <h3 className="text-lg font-semibold text-gray-900">{job.job_title}</h3>
                                {job.is_indian_tech_city && (
                                  <Badge variant="secondary" className="text-xs">
                                    🇮🇳 Indian Tech Hub
                                  </Badge>
                                )}
                                {job.application_status && (
                                  <Badge className={`text-xs ${getApplicationStatusColor(job.application_status)}`}>
                                    {formatApplicationStatus(job.application_status)}
                                  </Badge>
                                )}
                              </div>
                              <div className="flex items-center space-x-4 text-sm text-gray-600 mb-3">
                                <div className="flex items-center space-x-1">
                                  <Building className="w-4 h-4" />
                                  <span>{job.company_name}</span>
                                </div>
                                <div className="flex items-center space-x-1">
                                  <MapPin className="w-4 h-4" />
                                  <span>{job.location}</span>
                                </div>
                                {job.salary_range && (
                                  <div className="flex items-center space-x-1 text-green-600">
                                    <DollarSign className="w-4 h-4" />
                                    <span>{jobRecommendationService.formatIndianSalary(job.salary_range)}</span>
                                  </div>
                                )}
                              </div>
                            </div>
                            <div className="flex flex-col items-end space-y-2">
                              <Badge className={`${getMatchColor(job.match_score)} border`}>
                                {Math.round(job.match_score * 100)}% match
                              </Badge>
                              {job.market_demand && (
                                <Badge className={`text-xs ${jobRecommendationService.getMarketDemandColor(job.market_demand)}`}>
                                  {job.market_demand} demand
                                </Badge>
                              )}
                            </div>
                          </div>

                          {/* Match Score Progress */}
                          <div className="mb-4">
                            <div className="flex items-center justify-between text-sm mb-1">
                              <span className="text-gray-600">Match Score</span>
                              <span className="font-medium">{Math.round(job.match_score * 100)}%</span>
                            </div>
                            <Progress value={job.match_score * 100} className="h-2" />
                          </div>

                          {/* Skills Overview */}
                          <div className="flex items-center justify-between text-sm mb-4">
                            <div className="flex items-center space-x-4">
                              <div className="flex items-center space-x-1 text-green-600">
                                <CheckCircle className="w-4 h-4" />
                                <span>{job.skill_matches.length} skills match</span>
                              </div>
                              {job.skill_gaps.length > 0 && (
                                <div className="flex items-center space-x-1 text-orange-600">
                                  <AlertCircle className="w-4 h-4" />
                                  <span>{job.skill_gaps.length} gaps to fill</span>
                                </div>
                              )}
                            </div>
                            <div className="flex items-center space-x-1 text-gray-500">
                              <Clock className="w-4 h-4" />
                              <span>{job.experience_level || 'Not specified'}</span>
                            </div>
                          </div>

                          {/* Action Buttons */}
                          <div className="flex items-center justify-between pt-4 border-t border-gray-100">
                            <div className="flex items-center space-x-2">
                              <Button
                                variant="ghost"
                                size="sm"
                                onClick={() => handleJobInteraction(job, 'viewed')}
                                className="flex items-center space-x-1"
                              >
                                <Eye className="w-4 h-4" />
                                <span>View Details</span>
                              </Button>
                              <Button
                                variant="ghost"
                                size="sm"
                                onClick={() => {
                                  setSelectedJob(job);
                                  setShowGapAnalysis(true);
                                }}
                                className="flex items-center space-x-1"
                              >
                                <BarChart3 className="w-4 h-4" />
                                <span>Gap Analysis</span>
                              </Button>
                            </div>
                            <div className="flex items-center space-x-2">
                              {!job.application_status && (
                                <>
                                  <Button
                                    variant="ghost"
                                    size="sm"
                                    onClick={() => handleJobInteraction(job, 'interested')}
                                    className="flex items-center space-x-1"
                                  >
                                    <BookmarkPlus className="w-4 h-4" />
                                    <span>Save</span>
                                  </Button>
                                  <Button
                                    size="sm"
                                    onClick={() => handleApplyExternal(job)}
                                    className="flex items-center space-x-1"
                                  >
                                    <Send className="w-4 h-4" />
                                    <span>Apply</span>
                                  </Button>
                                </>
                              )}
                              {job.job_url && (
                                <Button
                                  variant="outline"
                                  size="sm"
                                  onClick={() => window.open(job.job_url, '_blank')}
                                  className="flex items-center space-x-1"
                                >
                                  <ExternalLink className="w-4 h-4" />
                                </Button>
                              )}
                            </div>
                          </div>

                          {/* Recommendation Reason */}
                          {job.recommendation_reason && (
                            <div className="mt-4 p-3 bg-blue-50 rounded-lg">
                              <p className="text-sm text-blue-800">
                                <strong>Why this matches:</strong> {job.recommendation_reason}
                              </p>
                            </div>
                          )}
                        </CardContent>
                      </Card>
                    </motion.div>
                  ))}
                </AnimatePresence>
              </div>
            )}
          </div>
        </TabsContent>

        {/* Applications Tab */}
        <TabsContent value="applications">
          <JobApplicationTracker />
        </TabsContent>

        {/* Market Insights Tab */}
        <TabsContent value="insights">
          <JobMarketInsights 
            targetRole={targetRole}
            preferredCities={preferredCities}
          />
        </TabsContent>
      </Tabs>

      {/* Gap Analysis Modal */}
      <AnimatePresence>
        {showGapAnalysis && selectedJob && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4"
            onClick={() => setShowGapAnalysis(false)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              className="bg-white rounded-lg max-w-2xl w-full max-h-[80vh] overflow-y-auto"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-xl font-semibold">Gap Analysis: {selectedJob.job_title}</h3>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setShowGapAnalysis(false)}
                  >
                    ×
                  </Button>
                </div>

                {selectedJob.gap_analysis && (
                  <div className="space-y-6">
                    {/* Skill Strength Overview */}
                    <div>
                      <h4 className="font-medium mb-3">Skill Alignment</h4>
                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <div className="flex items-center justify-between text-sm mb-1">
                            <span>Skill Strength</span>
                            <span className="font-medium">
                              {Math.round(selectedJob.gap_analysis.skill_strength * 100)}%
                            </span>
                          </div>
                          <Progress value={selectedJob.gap_analysis.skill_strength * 100} className="h-2" />
                        </div>
                        <div>
                          <div className="flex items-center justify-between text-sm mb-1">
                            <span>Experience Match</span>
                            <span className="font-medium">
                              {Math.round((1 - selectedJob.gap_analysis.experience_gap) * 100)}%
                            </span>
                          </div>
                          <Progress value={(1 - selectedJob.gap_analysis.experience_gap) * 100} className="h-2" />
                        </div>
                      </div>
                    </div>

                    {/* Skills Breakdown */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      {/* Matching Skills */}
                      {selectedJob.gap_analysis.strength_areas.length > 0 && (
                        <div>
                          <h5 className="font-medium text-green-700 mb-2 flex items-center">
                            <CheckCircle className="w-4 h-4 mr-1" />
                            Your Strengths
                          </h5>
                          <div className="space-y-1">
                            {selectedJob.gap_analysis.strength_areas.map((skill, idx) => (
                              <Badge key={idx} variant="secondary" className="bg-green-50 text-green-700 mr-1 mb-1">
                                {skill}
                              </Badge>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Skills to Develop */}
                      {selectedJob.gap_analysis.improvement_priority.length > 0 && (
                        <div>
                          <h5 className="font-medium text-orange-700 mb-2 flex items-center">
                            <AlertCircle className="w-4 h-4 mr-1" />
                            Priority Skills to Develop
                          </h5>
                          <div className="space-y-1">
                            {selectedJob.gap_analysis.improvement_priority.map((skill, idx) => (
                              <Badge key={idx} variant="outline" className="border-orange-200 text-orange-700 mr-1 mb-1">
                                {skill}
                              </Badge>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>

                    {/* Recommendations */}
                    <div className="bg-blue-50 rounded-lg p-4">
                      <h5 className="font-medium text-blue-900 mb-2">Recommendations</h5>
                      <ul className="text-sm text-blue-800 space-y-1">
                        <li>• Focus on developing the priority skills listed above</li>
                        <li>• Consider taking online courses or certifications</li>
                        <li>• Build projects that demonstrate these skills</li>
                        <li>• Update your resume to highlight matching strengths</li>
                      </ul>
                    </div>
                  </div>
                )}
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default EnhancedJobRecommendations;