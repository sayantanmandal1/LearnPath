/**
 * API service for making authenticated requests to the backend
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

class ApiService {
  constructor() {
    this.baseURL = API_BASE_URL;
  }

  getAuthHeaders() {
    const token = localStorage.getItem('access_token');
    return token ? { Authorization: `Bearer ${token}` } : {};
  }

  async request(endpoint, options = {}) {
    const url = `${this.baseURL}${endpoint}`;
    const config = {
      headers: {
        'Content-Type': 'application/json',
        ...this.getAuthHeaders(),
        ...options.headers,
      },
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
              config.headers.Authorization = `Bearer ${tokens.access_token}`;
              const retryResponse = await fetch(url, config);
              const retryData = await retryResponse.json();

              if (!retryResponse.ok) {
                throw new Error(retryData.detail || retryData.message || 'Request failed');
              }

              return retryData;
            }
          } catch (refreshError) {
            // Refresh failed, clear tokens and redirect to login
            localStorage.removeItem('access_token');
            localStorage.removeItem('refresh_token');
            window.location.href = '/login';
            throw new Error('Session expired. Please log in again.');
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
  async getProfile(userId) {
    return this.request(`/api/v1/profiles/${userId}`);
  }

  async createProfile(profileData) {
    return this.request('/api/v1/profiles', {
      method: 'POST',
      body: JSON.stringify(profileData),
    });
  }

  async updateProfile(userId, profileData) {
    return this.request(`/api/v1/profiles/${userId}`, {
      method: 'PUT',
      body: JSON.stringify(profileData),
    });
  }

  // Career analysis endpoints
  async analyzeCareer(analysisData) {
    return this.request('/api/v1/recommendations/analyze', {
      method: 'POST',
      body: JSON.stringify(analysisData),
    });
  }

  async getRecommendations(userId) {
    return this.request(`/api/v1/recommendations/${userId}`);
  }

  async getCareerTrajectory(userId) {
    return this.request(`/api/v1/career-trajectory/${userId}`);
  }

  async getLearningPaths(userId) {
    return this.request(`/api/v1/learning-paths/${userId}`);
  }

  // Job market endpoints
  async getJobMarketData(filters = {}) {
    const queryParams = new URLSearchParams(filters).toString();
    return this.request(`/api/v1/job-market/data${queryParams ? `?${queryParams}` : ''}`);
  }

  async searchJobs(query) {
    return this.request('/api/v1/job-market/search', {
      method: 'POST',
      body: JSON.stringify(query),
    });
  }

  // Analytics endpoints
  async getAnalytics(userId) {
    return this.request(`/api/v1/analytics/${userId}`);
  }

  async generateReport(userId, reportType) {
    return this.request(`/api/v1/analytics/${userId}/report`, {
      method: 'POST',
      body: JSON.stringify({ type: reportType }),
    });
  }
}

export const apiService = new ApiService();
export default apiService;