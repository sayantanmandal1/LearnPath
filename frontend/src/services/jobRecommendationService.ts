/**
 * Job recommendation service with application tracking
 */

import { apiClient } from './api';

export interface JobApplication {
  id: string;
  user_id: string;
  job_posting_id: string;
  job_title: string;
  company_name: string;
  job_url?: string;
  status: 'interested' | 'applied' | 'interviewing' | 'rejected' | 'accepted' | 'withdrawn';
  applied_date?: string;
  last_updated: string;
  match_score?: number;
  skill_matches?: string[];
  skill_gaps?: string[];
  application_method?: string;
  cover_letter?: string;
  notes?: string;
  interview_scheduled: boolean;
  interview_date?: string;
  interview_notes?: string;
  feedback_received: boolean;
  feedback_text?: string;
  rejection_reason?: string;
  created_at: string;
  updated_at: string;
}

export interface EnhancedJobMatch {
  job_posting_id: string;
  job_title: string;
  company_name: string;
  location: string;
  job_url?: string;
  match_score: number;
  skill_matches: string[];
  skill_gaps: string[];
  salary_range?: string;
  experience_level?: string;
  posted_date?: string;
  source: string;
  application_status?: string;
  application_id?: string;
  gap_analysis?: {
    skill_strength: number;
    total_skills_required: number;
    skills_matched: number;
    skills_missing: number;
    experience_gap: number;
    improvement_priority: string[];
    strength_areas: string[];
  };
  recommendation_reason: string;
  location_score: number;
  is_indian_tech_city: boolean;
  market_demand?: string;
  competition_level?: string;
}

export interface JobApplicationStats {
  total_applications: number;
  status_breakdown: { [key: string]: number };
  average_match_score?: number;
  applications_this_month: number;
  interviews_scheduled: number;
  success_rate: number;
  top_companies: Array<{ company: string; applications: number }>;
  application_timeline: Array<{ month: string; applications: number }>;
}

export interface LocationBasedJobSearch {
  target_role: string;
  preferred_cities?: string[];
  max_distance_km?: number;
  remote_acceptable?: boolean;
  hybrid_acceptable?: boolean;
  salary_min?: number;
  salary_max?: number;
  experience_level?: string;
  limit?: number;
}

export interface JobApplicationFeedback {
  feedback_type: 'recommendation_quality' | 'match_accuracy' | 'application_outcome' | 'general';
  rating?: number;
  feedback_text?: string;
  match_accuracy_rating?: number;
  recommendation_helpfulness?: number;
  gap_analysis_accuracy?: number;
  suggested_improvements?: { [key: string]: any };
}

export interface JobRecommendationFeedback {
  job_posting_id: string;
  user_interested?: boolean;
  user_applied?: boolean;
  match_score_feedback?: 'too_high' | 'accurate' | 'too_low';
  skill_match_feedback?: 'accurate' | 'missing_skills' | 'wrong_skills';
  location_feedback?: 'good' | 'not_preferred' | 'wrong_location';
  feedback_text?: string;
  improvement_suggestions?: { [key: string]: any };
}

export interface IndianTechJobsResponse {
  jobs: EnhancedJobMatch[];
  total_count: number;
  location_distribution: { [key: string]: number };
  salary_insights: { [key: string]: any };
  market_trends: { [key: string]: any };
  search_metadata: { [key: string]: any };
}

class JobRecommendationService {
  private baseUrl = '/api/v1/job-applications';

  /**
   * Create a new job application
   */
  async createApplication(applicationData: {
    job_posting_id: string;
    job_title: string;
    company_name: string;
    job_url?: string;
    match_score?: number;
    skill_matches?: string[];
    skill_gaps?: string[];
    application_method?: string;
    cover_letter?: string;
    notes?: string;
  }): Promise<JobApplication> {
    const response = await apiClient.post(`${this.baseUrl}/applications`, applicationData);
    return response.data;
  }

  /**
   * Get user's job applications
   */
  async getUserApplications(params?: {
    status?: string;
    limit?: number;
    offset?: number;
  }): Promise<JobApplication[]> {
    const response = await apiClient.get(`${this.baseUrl}/applications`, { params });
    return response.data;
  }

  /**
   * Update a job application
   */
  async updateApplication(
    applicationId: string,
    updateData: Partial<JobApplication>
  ): Promise<JobApplication> {
    const response = await apiClient.put(`${this.baseUrl}/applications/${applicationId}`, updateData);
    return response.data;
  }

  /**
   * Get application statistics
   */
  async getApplicationStats(): Promise<JobApplicationStats> {
    const response = await apiClient.get(`${this.baseUrl}/applications/stats`);
    return response.data;
  }

  /**
   * Add feedback for a job application
   */
  async addApplicationFeedback(
    applicationId: string,
    feedback: JobApplicationFeedback
  ): Promise<{ feedback_id: string; message: string }> {
    const response = await apiClient.post(
      `${this.baseUrl}/applications/${applicationId}/feedback`,
      feedback
    );
    return response.data;
  }

  /**
   * Add feedback for job recommendations
   */
  async addRecommendationFeedback(
    feedback: JobRecommendationFeedback
  ): Promise<{ feedback_id: string; message: string }> {
    const response = await apiClient.post(`${this.baseUrl}/recommendations/feedback`, feedback);
    return response.data;
  }

  /**
   * Mark a job as applied
   */
  async markJobAsApplied(
    jobPostingId: string,
    applicationMethod: string = 'external'
  ): Promise<{ application_id: string; message: string }> {
    const response = await apiClient.post(
      `${this.baseUrl}/applications/mark-applied/${jobPostingId}`,
      null,
      { params: { application_method: applicationMethod } }
    );
    return response.data;
  }

  /**
   * Get enhanced job recommendations with application tracking
   */
  async getEnhancedRecommendations(
    searchParams: LocationBasedJobSearch
  ): Promise<EnhancedJobMatch[]> {
    const response = await apiClient.post(`${this.baseUrl}/enhanced-recommendations`, searchParams);
    return response.data;
  }

  /**
   * Get Indian tech jobs with tracking and insights
   */
  async getIndianTechJobs(params: {
    target_role: string;
    preferred_cities?: string[];
    salary_min?: number;
    salary_max?: number;
    experience_level?: string;
    limit?: number;
  }): Promise<IndianTechJobsResponse> {
    const response = await apiClient.get(`${this.baseUrl}/indian-tech-jobs`, { params });
    return response.data;
  }

  /**
   * Refresh job recommendations
   */
  async refreshRecommendations(params: {
    target_role: string;
    preferred_cities?: string[];
  }): Promise<{ message: string; target_role: string; cities: string[] }> {
    const response = await apiClient.post(`${this.baseUrl}/refresh-recommendations`, null, { params });
    return response.data;
  }

  /**
   * Get application insights and analytics
   */
  async getApplicationInsights(): Promise<{
    stats: JobApplicationStats;
    insights: { [key: string]: any };
    recommendations: string[];
  }> {
    const response = await apiClient.get(`${this.baseUrl}/application-insights`);
    return response.data;
  }

  /**
   * Get job application status for a specific job
   */
  async getJobApplicationStatus(jobPostingId: string): Promise<JobApplication | null> {
    try {
      const applications = await this.getUserApplications();
      return applications.find(app => app.job_posting_id === jobPostingId) || null;
    } catch (error) {
      console.error('Error getting job application status:', error);
      return null;
    }
  }

  /**
   * Track job recommendation interaction
   */
  async trackRecommendationInteraction(
    jobPostingId: string,
    interactionType: 'viewed' | 'clicked' | 'interested' | 'not_interested'
  ): Promise<void> {
    try {
      const feedback: JobRecommendationFeedback = {
        job_posting_id: jobPostingId,
        user_interested: interactionType === 'interested' ? true : 
                        interactionType === 'not_interested' ? false : undefined,
      };
      
      await this.addRecommendationFeedback(feedback);
    } catch (error) {
      console.error('Error tracking recommendation interaction:', error);
    }
  }

  /**
   * Get location-based job recommendations
   */
  async getLocationBasedJobs(
    targetRole: string,
    preferredCities: string[],
    options?: {
      remoteAcceptable?: boolean;
      hybridAcceptable?: boolean;
      salaryMin?: number;
      salaryMax?: number;
      experienceLevel?: string;
      limit?: number;
    }
  ): Promise<EnhancedJobMatch[]> {
    const searchParams: LocationBasedJobSearch = {
      target_role: targetRole,
      preferred_cities: preferredCities,
      remote_acceptable: options?.remoteAcceptable ?? false,
      hybrid_acceptable: options?.hybridAcceptable ?? true,
      salary_min: options?.salaryMin,
      salary_max: options?.salaryMax,
      experience_level: options?.experienceLevel,
      limit: options?.limit ?? 50,
    };

    return this.getEnhancedRecommendations(searchParams);
  }

  /**
   * Get job market insights for Indian tech cities
   */
  async getIndianTechMarketInsights(role: string, cities?: string[]): Promise<{ [key: string]: any }> {
    try {
      const response = await apiClient.get('/api/v1/real-time-jobs/market-insights', {
        params: { role, cities }
      });
      return response.data;
    } catch (error) {
      console.error('Error getting market insights:', error);
      return {};
    }
  }

  /**
   * Format salary for Indian market
   */
  formatIndianSalary(salaryRange?: string): string {
    if (!salaryRange) return 'Not disclosed';
    
    // Convert common formats to Indian Lakh notation
    return salaryRange
      .replace(/(\d+)k/g, '₹$1,000')
      .replace(/(\d+)L/g, '₹$1 Lakh')
      .replace(/(\d+) LPA/g, '₹$1 LPA');
  }

  /**
   * Get Indian tech cities list
   */
  getIndianTechCities(): string[] {
    return [
      'Bangalore',
      'Hyderabad', 
      'Pune',
      'Chennai',
      'Mumbai',
      'Delhi NCR',
      'Kolkata',
      'Ahmedabad',
      'Kochi',
      'Indore',
      'Jaipur',
      'Chandigarh'
    ];
  }

  /**
   * Check if location is an Indian tech city
   */
  isIndianTechCity(location: string): boolean {
    const indianCities = this.getIndianTechCities().map(city => city.toLowerCase());
    const locationLower = location.toLowerCase();
    
    return indianCities.some(city => 
      locationLower.includes(city) || 
      locationLower.includes(city.replace(' ', ''))
    );
  }

  /**
   * Get competition level color
   */
  getCompetitionLevelColor(level?: string): string {
    switch (level) {
      case 'low': return 'text-green-600 bg-green-50';
      case 'medium': return 'text-yellow-600 bg-yellow-50';
      case 'high': return 'text-red-600 bg-red-50';
      default: return 'text-gray-600 bg-gray-50';
    }
  }

  /**
   * Get market demand color
   */
  getMarketDemandColor(demand?: string): string {
    switch (demand) {
      case 'high': return 'text-green-600 bg-green-50';
      case 'medium': return 'text-blue-600 bg-blue-50';
      case 'low': return 'text-orange-600 bg-orange-50';
      default: return 'text-gray-600 bg-gray-50';
    }
  }
}

export const jobRecommendationService = new JobRecommendationService();
export default jobRecommendationService;