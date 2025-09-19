/**
 * API service for making authenticated requests to the backend
 * Updated to work with backend JWT authentication instead of Supabase for data operations
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

// TypeScript interfaces for API service
interface RequestOptions {
  method?: string;
  headers?: Record<string, string>;
  body?: string | FormData;
}

interface DashboardOptions {
  includeJobMatches?: boolean;
  includeSkillRadar?: boolean;
  includeCareerProgress?: boolean;
}

interface SkillRadarOptions {
  includeMarketComparison?: boolean;
  skillCategories?: string[];
}

interface CareerProgressOptions {
  trackingPeriodDays?: number;
  includePredictions?: boolean;
  includeMilestones?: boolean;
}

interface JobRecommendationOptions {
  limit?: number;
  preferredCities?: string[];
  targetRole?: string | null;
  minMatchScore?: number;
  includeMarketInsights?: boolean;
}

interface AnalysisOptions {
  forceRefresh?: boolean;
  analysisTypes?: string[];
}

interface MarketInsightsOptions {
  role?: string | null;
  location?: string;
  skillCategory?: string;
}

interface JobMarketTrendsOptions {
  timePeriod?: string;
  includeSalaryTrends?: boolean;
  includeSkillDemand?: boolean;
  location?: string;
}

interface RealTimeJobsOptions {
  limit?: number;
  locationFilter?: string;
  role?: string;
  cities?: string[];
}

interface JobMatchingOptions {
  minMatchScore?: number;
  includeGapAnalysis?: boolean;
}

class ApiService {
  private baseURL: string;

  constructor() {
    this.baseURL = API_BASE_URL;
  }

  private getAuthHeaders(): Record<string, string> {
    const token = localStorage.getItem('access_token');
    return token ? { Authorization: `Bearer ${token}` } : {};
  }

  async request(endpoint: string, options: RequestOptions = {}): Promise<any> {
    const url = `${this.baseURL}${endpoint}`;
    
    // Don't set Content-Type for FormData, let browser handle it
    const isFormData = options.body instanceof FormData;
    const headers = {
      ...(isFormData ? {} : { 'Content-Type': 'application/json' }),
      ...this.getAuthHeaders(),
      ...options.headers,
    };
    
    const config: RequestInit = {
      headers,
      ...options,
    };

    try {
      const response = await fetch(url, config);
      
      // Handle 401 responses by attempting token refresh
      if (response.status === 401) {
        const refreshToken = localStorage.getItem('refresh_token');
        if (refreshToken) {
          try {
            // Try to refresh the token
            const refreshResponse = await fetch(`${this.baseURL}/api/v1/auth/refresh`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ refresh_token: refreshToken }),
            });

            if (refreshResponse.ok) {
              const tokens = await refreshResponse.json();
              localStorage.setItem('access_token', tokens.access_token);
              localStorage.setItem('refresh_token', tokens.refresh_token);

              // Retry the original request with new token
              config.headers = {
                ...config.headers,
                Authorization: `Bearer ${tokens.access_token}`
              };
              const retryResponse = await fetch(url, config);
              const retryData = await retryResponse.json();

              if (!retryResponse.ok) {
                throw new Error(retryData.detail || retryData.message || 'Request failed');
              }

              return retryData;
            }
          } catch (refreshError) {
            // Refresh failed, clear tokens but don't redirect automatically
            localStorage.removeItem('access_token');
            localStorage.removeItem('refresh_token');
            throw new Error('Not authenticated');
          }
        }
      }

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || data.message || 'Request failed');
      }

      return data;
    } catch (error) {
      console.error('API request failed:', error);
      throw error;
    }
  }

  // Profile endpoints
  async getProfile(userId: string): Promise<any> {
    return this.request(`/api/v1/profiles/${userId}`);
  }

  async createProfile(profileData: any): Promise<any> {
    return this.request('/api/v1/profiles', {
      method: 'POST',
      body: JSON.stringify(profileData),
    });
  }

  async updateProfile(userId: string, profileData: any): Promise<any> {
    return this.request(`/api/v1/profiles/${userId}`, {
      method: 'PUT',
      body: JSON.stringify(profileData),
    });
  }

  // Career analysis endpoints
  async analyzeCareer(analysisData: any): Promise<any> {
    return this.request('/api/v1/recommendations/analyze', {
      method: 'POST',
      body: JSON.stringify(analysisData),
    });
  }

  async getRecommendations(userId: string): Promise<any> {
    return this.request(`/api/v1/recommendations/${userId}`);
  }

  async getCareerTrajectory(userId: string): Promise<any> {
    return this.request(`/api/v1/career-trajectory/${userId}`);
  }

  async getLearningPaths(userId: string): Promise<any> {
    return this.request(`/api/v1/learning-paths/${userId}`);
  }

  // Job market endpoints
  async getJobMarketData(filters: Record<string, any> = {}): Promise<any> {
    const queryParams = new URLSearchParams(filters).toString();
    return this.request(`/api/v1/job-market/data${queryParams ? `?${queryParams}` : ''}`);
  }

  async searchJobs(query: any): Promise<any> {
    return this.request('/api/v1/job-market/search', {
      method: 'POST',
      body: JSON.stringify(query),
    });
  }

  // Analytics endpoints
  async getAnalytics(userId: string): Promise<any> {
    return this.request(`/api/v1/analytics/${userId}`);
  }

  async generateReport(userId: string, reportType: string): Promise<any> {
    return this.request(`/api/v1/analytics/${userId}/report`, {
      method: 'POST',
      body: JSON.stringify({ type: reportType }),
    });
  }

  // Resume endpoints
  async uploadResume(file: File): Promise<any> {
    const formData = new FormData();
    formData.append('file', file);
    
    return this.request('/api/v1/resume/upload', {
      method: 'POST',
      body: formData,
      headers: {
        // Remove Content-Type to let browser set it with boundary for FormData
        ...this.getAuthHeaders(),
      },
    });
  }

  async getResumeData(userId: string): Promise<any> {
    return this.request(`/api/v1/resume/${userId}`);
  }

  async updateResumeData(userId: string, resumeData: any): Promise<any> {
    return this.request(`/api/v1/resume/${userId}`, {
      method: 'PUT',
      body: JSON.stringify(resumeData),
    });
  }

  // Platform connection endpoints
  async getPlatformAccounts(userId: string): Promise<any> {
    return this.request(`/api/v1/platforms/${userId}/accounts`);
  }

  async createPlatformAccount(accountData: any): Promise<any> {
    return this.request('/api/v1/platforms/accounts', {
      method: 'POST',
      body: JSON.stringify(accountData),
    });
  }

  async updatePlatformAccount(accountId: string, accountData: any): Promise<any> {
    return this.request(`/api/v1/platforms/accounts/${accountId}`, {
      method: 'PUT',
      body: JSON.stringify(accountData),
    });
  }

  async deletePlatformAccount(accountId: string): Promise<any> {
    return this.request(`/api/v1/platforms/accounts/${accountId}`, {
      method: 'DELETE',
    });
  }

  async validatePlatformAccount(validationData: any): Promise<any> {
    return this.request('/api/v1/external-profiles/validate', {
      method: 'POST',
      body: JSON.stringify(validationData),
    });
  }

  async extractPlatformData(extractionData: any): Promise<any> {
    return this.request('/api/v1/external-profiles/extract', {
      method: 'POST',
      body: JSON.stringify(extractionData),
    });
  }

  async refreshPlatformData(accountId: string): Promise<any> {
    return this.request(`/api/v1/platforms/accounts/${accountId}/refresh`, {
      method: 'POST',
    });
  }

  async getPlatformDataPreview(accountId: string): Promise<any> {
    return this.request(`/api/v1/platforms/accounts/${accountId}/preview`);
  }

  // Enhanced Dashboard endpoints for real-time data integration
  async getDashboardSummary(): Promise<any> {
    return this.request('/api/v1/dashboard/summary');
  }

  async getComprehensiveDashboardData(options: DashboardOptions = {}): Promise<any> {
    const queryParams = new URLSearchParams({
      include_job_matches: String(options.includeJobMatches !== false),
      include_skill_radar: String(options.includeSkillRadar !== false),
      include_career_progress: String(options.includeCareerProgress !== false),
    }).toString();
    return this.request(`/api/v1/dashboard/comprehensive-data?${queryParams}`);
  }

  async getSkillRadarData(options: SkillRadarOptions = {}): Promise<any> {
    const queryParams = new URLSearchParams();
    if (options.includeMarketComparison !== false) {
      queryParams.append('include_market_comparison', 'true');
    }
    if (options.skillCategories) {
      queryParams.append('skill_categories', options.skillCategories.join(','));
    }
    return this.request(`/api/v1/dashboard/skill-radar?${queryParams.toString()}`);
  }

  async getCareerProgressTracking(options: CareerProgressOptions = {}): Promise<any> {
    const queryParams = new URLSearchParams({
      tracking_period_days: String(options.trackingPeriodDays || 90),
      include_predictions: String(options.includePredictions !== false),
      include_milestones: String(options.includeMilestones !== false),
    }).toString();
    return this.request(`/api/v1/dashboard/career-progress?${queryParams}`);
  }

  async getPersonalizedJobRecommendations(options: JobRecommendationOptions = {}): Promise<any> {
    const queryParams = new URLSearchParams({
      limit: String(options.limit || 20),
      min_match_score: String(options.minMatchScore || 0.6),
      include_market_insights: String(options.includeMarketInsights !== false),
    });
    if (options.preferredCities) {
      queryParams.append('preferred_cities', options.preferredCities.join(','));
    }
    if (options.targetRole) {
      queryParams.append('target_role', options.targetRole);
    }
    return this.request(`/api/v1/dashboard/job-recommendations?${queryParams.toString()}`);
  }

  async getRealTimeAnalysisResults(options: AnalysisOptions = {}): Promise<any> {
    const queryParams = new URLSearchParams({
      force_refresh: String(options.forceRefresh || false),
    });
    if (options.analysisTypes) {
      queryParams.append('analysis_types', options.analysisTypes.join(','));
    }
    return this.request(`/api/v1/dashboard/real-time-analysis?${queryParams.toString()}`);
  }

  async getPersonalizedContent(): Promise<any> {
    return this.request('/api/v1/dashboard/personalized-content');
  }

  async getDashboardQuickStats(): Promise<any> {
    return this.request('/api/v1/dashboard/quick-stats');
  }

  // Market insights endpoints
  async getMarketInsights(options: MarketInsightsOptions = {}): Promise<any> {
    const queryParams = new URLSearchParams();
    if (options.role) {
      queryParams.append('role', options.role);
    }
    if (options.location) {
      queryParams.append('location', options.location);
    }
    if (options.skillCategory) {
      queryParams.append('skill_category', options.skillCategory);
    }
    return this.request(`/api/v1/market-insights?${queryParams.toString()}`);
  }

  async getJobMarketTrends(options: JobMarketTrendsOptions = {}): Promise<any> {
    const queryParams = new URLSearchParams({
      time_period: options.timePeriod || '6months',
      include_salary_trends: String(options.includeSalaryTrends !== false),
      include_skill_demand: String(options.includeSkillDemand !== false),
    });
    if (options.location) {
      queryParams.append('location', options.location);
    }
    return this.request(`/api/v1/market-insights/trends?${queryParams.toString()}`);
  }

  // Real-time job service endpoints
  async getRealTimeJobs(options: RealTimeJobsOptions = {}): Promise<any> {
    const queryParams = new URLSearchParams({
      limit: String(options.limit || 20),
      location_filter: options.locationFilter || 'india',
    });
    if (options.role) {
      queryParams.append('role', options.role);
    }
    if (options.cities) {
      queryParams.append('cities', options.cities.join(','));
    }
    return this.request(`/api/v1/real-time-jobs?${queryParams.toString()}`);
  }

  async getJobMatchingResults(options: JobMatchingOptions = {}): Promise<any> {
    const queryParams = new URLSearchParams({
      min_match_score: String(options.minMatchScore || 0.6),
      include_gap_analysis: String(options.includeGapAnalysis !== false),
    });
    return this.request(`/api/v1/real-time-jobs/matches?${queryParams.toString()}`);
  }
}

export const apiService = new ApiService();
export default apiService;