/**
 * Job Application Tracker Component
 * Tracks and displays user's job applications with status updates
 */
import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Badge } from '../ui/badge';
import { Button } from '../ui/button';
import { Progress } from '../ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../ui/tabs';
import { 
  Building, 
  MapPin, 
  Calendar, 
  Clock, 
  CheckCircle, 
  XCircle, 
  AlertCircle,
  TrendingUp,
  Users,
  Star,
  Edit,
  Trash2,
  ExternalLink,
  MessageSquare,
  BarChart3,
  Target
} from 'lucide-react';

import { 
  jobRecommendationService, 
  JobApplication, 
  JobApplicationStats,
  JobApplicationFeedback 
} from '../../services/jobRecommendationService';

interface JobApplicationTrackerProps {
  className?: string;
}

export const JobApplicationTracker: React.FC<JobApplicationTrackerProps> = ({
  className = '',
}) => {
  const [applications, setApplications] = useState<JobApplication[]>([]);
  const [stats, setStats] = useState<JobApplicationStats | null>(null);
  const [loading, setLoading] = useState(false);
  const [selectedApplication, setSelectedApplication] = useState<JobApplication | null>(null);
  const [showFeedbackModal, setShowFeedbackModal] = useState(false);
  const [activeTab, setActiveTab] = useState('all');
  const [feedbackForm, setFeedbackForm] = useState<Partial<JobApplicationFeedback>>({});

  useEffect(() => {
    loadApplications();
    loadStats();
  }, []);

  const loadApplications = async (status?: string) => {
    setLoading(true);
    try {
      const apps = await jobRecommendationService.getUserApplications({
        status: status === 'all' ? undefined : status,
        limit: 100
      });
      setApplications(apps);
    } catch (error) {
      console.error('Error loading applications:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadStats = async () => {
    try {
      const applicationStats = await jobRecommendationService.getApplicationStats();
      setStats(applicationStats);
    } catch (error) {
      console.error('Error loading stats:', error);
    }
  };

  const handleStatusUpdate = async (applicationId: string, newStatus: string) => {
    try {
      await jobRecommendationService.updateApplication(applicationId, {
        status: newStatus,
        last_updated: new Date().toISOString()
      });
      
      // Refresh data
      await loadApplications();
      await loadStats();
    } catch (error) {
      console.error('Error updating application status:', error);
    }
  };

  const handleAddFeedback = async () => {
    if (!selectedApplication || !feedbackForm.feedback_type) return;
    
    try {
      await jobRecommendationService.addApplicationFeedback(
        selectedApplication.id,
        feedbackForm as JobApplicationFeedback
      );
      
      setShowFeedbackModal(false);
      setFeedbackForm({});
      setSelectedApplication(null);
      
      // Refresh data
      await loadApplications();
    } catch (error) {
      console.error('Error adding feedback:', error);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'interested': return 'text-blue-600 bg-blue-50 border-blue-200';
      case 'applied': return 'text-green-600 bg-green-50 border-green-200';
      case 'interviewing': return 'text-purple-600 bg-purple-50 border-purple-200';
      case 'rejected': return 'text-red-600 bg-red-50 border-red-200';
      case 'accepted': return 'text-emerald-600 bg-emerald-50 border-emerald-200';
      case 'withdrawn': return 'text-gray-600 bg-gray-50 border-gray-200';
      default: return 'text-gray-600 bg-gray-50 border-gray-200';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'interested': return <Star className="w-4 h-4" />;
      case 'applied': return <CheckCircle className="w-4 h-4" />;
      case 'interviewing': return <Users className="w-4 h-4" />;
      case 'rejected': return <XCircle className="w-4 h-4" />;
      case 'accepted': return <CheckCircle className="w-4 h-4" />;
      case 'withdrawn': return <AlertCircle className="w-4 h-4" />;
      default: return <Clock className="w-4 h-4" />;
    }
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-IN', {
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    });
  };

  const getApplicationsByStatus = (status: string) => {
    if (status === 'all') return applications;
    return applications.filter(app => app.status === status);
  };

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Header with Stats */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">Application Tracker</h2>
          <p className="text-gray-600">
            Track your job applications and their progress
          </p>
        </div>
        {stats && (
          <div className="flex items-center space-x-6 text-sm">
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">{stats.total_applications}</div>
              <div className="text-gray-600">Total</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">{stats.success_rate}%</div>
              <div className="text-gray-600">Success Rate</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-purple-600">{stats.interviews_scheduled}</div>
              <div className="text-gray-600">Interviews</div>
            </div>
          </div>
        )}
      </div>

      {/* Stats Cards */}
      {stats && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600">This Month</p>
                  <p className="text-2xl font-bold">{stats.applications_this_month}</p>
                </div>
                <TrendingUp className="w-8 h-8 text-blue-500" />
              </div>
            </CardContent>
          </Card>
          
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600">Avg Match Score</p>
                  <p className="text-2xl font-bold">
                    {stats.average_match_score ? Math.round(stats.average_match_score * 100) : 0}%
                  </p>
                </div>
                <Target className="w-8 h-8 text-green-500" />
              </div>
            </CardContent>
          </Card>
          
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600">In Progress</p>
                  <p className="text-2xl font-bold">
                    {(stats.status_breakdown.applied || 0) + (stats.status_breakdown.interviewing || 0)}
                  </p>
                </div>
                <Clock className="w-8 h-8 text-orange-500" />
              </div>
            </CardContent>
          </Card>
          
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600">Accepted</p>
                  <p className="text-2xl font-bold">{stats.status_breakdown.accepted || 0}</p>
                </div>
                <CheckCircle className="w-8 h-8 text-emerald-500" />
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Application Tabs */}
      <Tabs value={activeTab} onValueChange={(value) => {
        setActiveTab(value);
        loadApplications(value);
      }}>
        <TabsList className="grid w-full grid-cols-6">
          <TabsTrigger value="all">All</TabsTrigger>
          <TabsTrigger value="interested">Interested</TabsTrigger>
          <TabsTrigger value="applied">Applied</TabsTrigger>
          <TabsTrigger value="interviewing">Interviewing</TabsTrigger>
          <TabsTrigger value="rejected">Rejected</TabsTrigger>
          <TabsTrigger value="accepted">Accepted</TabsTrigger>
        </TabsList>

        <TabsContent value={activeTab} className="space-y-4">
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
          ) : getApplicationsByStatus(activeTab).length === 0 ? (
            <Card>
              <CardContent className="text-center py-12">
                <BarChart3 className="w-12 h-12 mx-auto mb-4 text-gray-400" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">No applications found</h3>
                <p className="text-gray-600">
                  {activeTab === 'all' 
                    ? "You haven't applied to any jobs yet. Start exploring recommendations!"
                    : `No applications with status "${activeTab}"`
                  }
                </p>
              </CardContent>
            </Card>
          ) : (
            <div className="space-y-4">
              <AnimatePresence>
                {getApplicationsByStatus(activeTab).map((application, index) => (
                  <motion.div
                    key={application.id}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    transition={{ duration: 0.3, delay: index * 0.1 }}
                  >
                    <Card className="hover:shadow-lg transition-shadow">
                      <CardContent className="p-6">
                        <div className="flex items-start justify-between mb-4">
                          <div className="flex-1">
                            <div className="flex items-center space-x-2 mb-2">
                              <h3 className="text-lg font-semibold text-gray-900">
                                {application.job_title}
                              </h3>
                              <Badge className={`${getStatusColor(application.status)} border flex items-center space-x-1`}>
                                {getStatusIcon(application.status)}
                                <span>{application.status.charAt(0).toUpperCase() + application.status.slice(1)}</span>
                              </Badge>
                            </div>
                            
                            <div className="flex items-center space-x-4 text-sm text-gray-600 mb-3">
                              <div className="flex items-center space-x-1">
                                <Building className="w-4 h-4" />
                                <span>{application.company_name}</span>
                              </div>
                              <div className="flex items-center space-x-1">
                                <Calendar className="w-4 h-4" />
                                <span>Applied: {formatDate(application.created_at)}</span>
                              </div>
                              {application.match_score && (
                                <div className="flex items-center space-x-1">
                                  <Target className="w-4 h-4" />
                                  <span>{Math.round(application.match_score * 100)}% match</span>
                                </div>
                              )}
                            </div>
                          </div>
                        </div>

                        {/* Match Score Progress */}
                        {application.match_score && (
                          <div className="mb-4">
                            <div className="flex items-center justify-between text-sm mb-1">
                              <span className="text-gray-600">Match Score</span>
                              <span className="font-medium">{Math.round(application.match_score * 100)}%</span>
                            </div>
                            <Progress value={application.match_score * 100} className="h-2" />
                          </div>
                        )}

                        {/* Skills Overview */}
                        {(application.skill_matches || application.skill_gaps) && (
                          <div className="mb-4">
                            <div className="flex items-center space-x-4 text-sm">
                              {application.skill_matches && application.skill_matches.length > 0 && (
                                <div className="flex items-center space-x-1 text-green-600">
                                  <CheckCircle className="w-4 h-4" />
                                  <span>{application.skill_matches.length} skills match</span>
                                </div>
                              )}
                              {application.skill_gaps && application.skill_gaps.length > 0 && (
                                <div className="flex items-center space-x-1 text-orange-600">
                                  <AlertCircle className="w-4 h-4" />
                                  <span>{application.skill_gaps.length} gaps to fill</span>
                                </div>
                              )}
                            </div>
                          </div>
                        )}

                        {/* Notes */}
                        {application.notes && (
                          <div className="mb-4 p-3 bg-gray-50 rounded-lg">
                            <p className="text-sm text-gray-700">{application.notes}</p>
                          </div>
                        )}

                        {/* Action Buttons */}
                        <div className="flex items-center justify-between pt-4 border-t border-gray-100">
                          <div className="flex items-center space-x-2">
                            <select
                              value={application.status}
                              onChange={(e) => handleStatusUpdate(application.id, e.target.value)}
                              className="text-sm border border-gray-300 rounded px-2 py-1"
                            >
                              <option value="interested">Interested</option>
                              <option value="applied">Applied</option>
                              <option value="interviewing">Interviewing</option>
                              <option value="rejected">Rejected</option>
                              <option value="accepted">Accepted</option>
                              <option value="withdrawn">Withdrawn</option>
                            </select>
                          </div>
                          
                          <div className="flex items-center space-x-2">
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => {
                                setSelectedApplication(application);
                                setShowFeedbackModal(true);
                              }}
                              className="flex items-center space-x-1"
                            >
                              <MessageSquare className="w-4 h-4" />
                              <span>Feedback</span>
                            </Button>
                            
                            {application.job_url && (
                              <Button
                                variant="outline"
                                size="sm"
                                onClick={() => window.open(application.job_url, '_blank')}
                                className="flex items-center space-x-1"
                              >
                                <ExternalLink className="w-4 h-4" />
                                <span>View Job</span>
                              </Button>
                            )}
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  </motion.div>
                ))}
              </AnimatePresence>
            </div>
          )}
        </TabsContent>
      </Tabs>

      {/* Feedback Modal */}
      <AnimatePresence>
        {showFeedbackModal && selectedApplication && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4"
            onClick={() => setShowFeedbackModal(false)}
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              className="bg-white rounded-lg max-w-md w-full"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="p-6">
                <h3 className="text-xl font-semibold mb-4">
                  Add Feedback: {selectedApplication.job_title}
                </h3>
                
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Feedback Type
                    </label>
                    <select
                      value={feedbackForm.feedback_type || ''}
                      onChange={(e) => setFeedbackForm({...feedbackForm, feedback_type: e.target.value})}
                      className="w-full border border-gray-300 rounded px-3 py-2"
                    >
                      <option value="">Select type</option>
                      <option value="recommendation_quality">Recommendation Quality</option>
                      <option value="match_accuracy">Match Accuracy</option>
                      <option value="application_outcome">Application Outcome</option>
                      <option value="general">General</option>
                    </select>
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Rating (1-5)
                    </label>
                    <select
                      value={feedbackForm.rating || ''}
                      onChange={(e) => setFeedbackForm({...feedbackForm, rating: parseInt(e.target.value)})}
                      className="w-full border border-gray-300 rounded px-3 py-2"
                    >
                      <option value="">Select rating</option>
                      <option value="1">1 - Poor</option>
                      <option value="2">2 - Fair</option>
                      <option value="3">3 - Good</option>
                      <option value="4">4 - Very Good</option>
                      <option value="5">5 - Excellent</option>
                    </select>
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Comments
                    </label>
                    <textarea
                      value={feedbackForm.feedback_text || ''}
                      onChange={(e) => setFeedbackForm({...feedbackForm, feedback_text: e.target.value})}
                      className="w-full border border-gray-300 rounded px-3 py-2 h-24"
                      placeholder="Share your thoughts about this recommendation..."
                    />
                  </div>
                </div>
                
                <div className="flex items-center justify-end space-x-3 mt-6">
                  <Button
                    variant="outline"
                    onClick={() => setShowFeedbackModal(false)}
                  >
                    Cancel
                  </Button>
                  <Button
                    onClick={handleAddFeedback}
                    disabled={!feedbackForm.feedback_type}
                  >
                    Submit Feedback
                  </Button>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default JobApplicationTracker;