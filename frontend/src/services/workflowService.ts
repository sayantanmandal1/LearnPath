/**
 * Workflow Integration Service
 * 
 * This service handles the complete user workflow integration on the frontend,
 * coordinating between resume upload, platform connections, AI analysis, and dashboard updates.
 */

import { apiService } from './api';

export interface WorkflowProgress {
  user_id: string;
  current_stage: string;
  completed_stages: string[];
  progress_percentage: number;
  errors: string[];
  warnings: string[];
  started_at: string;
  updated_at: string;
  estimated_completion?: string;
}

export interface WorkflowResult {
  success: boolean;
  user_id: string;
  progress: WorkflowProgress;
  resume_data?: any;
  platform_data?: any;
  analysis_results?: any;
  dashboard_data?: any;
  job_matches?: any[];
  errors: string[];
  execution_time: number;
}

export interface ValidationResult {
  user_id: string;
  timestamp: string;
  checks: Record<string, any>;
  overall_status: string;
  issues: string[];
}

export interface PlatformAccount {
  platform: string;
  username?: string;
  profile_url?: string;
}

export interface WorkflowExecutionRequest {
  platform_accounts?: Record<string, Record<string, string>>;
  skip_resume?: boolean;
  skip_platforms?: boolean;
  skip_ai_analysis?: boolean;
}

class WorkflowService {
  private progressCallbacks: ((progress: WorkflowProgress) => void)[] = [];
  private currentProgress: WorkflowProgress | null = null;

  /**
   * Execute complete workflow with resume and platform accounts
   */
  async executeCompleteWorkflow(
    resumeFile?: File,
    platformAccounts?: Record<string, Record<string, string>>,
    options: Partial<WorkflowExecutionRequest> = {}
  ): Promise<WorkflowResult> {
    try {
      const formData = new FormData();
      
      // Add resume file if provided
      if (resumeFile && !options.skip_resume) {
        formData.append('resume_file', resumeFile);
      }
      
      // Add request data
      const requestData: WorkflowExecutionRequest = {
        platform_accounts: platformAccounts,
        skip_resume: options.skip_resume || !resumeFile,
        skip_platforms: options.skip_platforms || !platformAccounts,
        skip_ai_analysis: options.skip_ai_analysis || false
      };
      
      formData.append('request', JSON.stringify(requestData));
      
      const response = await apiService.request('/api/v1/workflow/execute', {
        method: 'POST',
        body: formData,
        // Don't set Content-Type header, let browser set it for FormData
        headers: {}
      });
      
      this.currentProgress = response.progress;
      this.notifyProgressCallbacks(response.progress);
      
      return response;
    } catch (error) {
      console.error('Workflow execution failed:', error);
      throw error;
    }
  }

  /**
   * Execute workflow step by step with progress tracking
   */
  async executeWorkflowWithProgress(
    resumeFile?: File,
    platformAccounts?: Record<string, Record<string, string>>,
    onProgress?: (progress: WorkflowProgress) => void
  ): Promise<WorkflowResult> {
    if (onProgress) {
      this.addProgressCallback(onProgress);
    }

    try {
      // Start workflow execution
      const result = await this.executeCompleteWorkflow(resumeFile, platformAccounts);
      
      // Poll for progress updates during execution
      const progressInterval = setInterval(async () => {
        try {
          const progress = await this.getWorkflowProgress();
          if (progress) {
            this.currentProgress = progress;
            this.notifyProgressCallbacks(progress);
            
            // Stop polling when completed
            if (progress.current_stage === 'completed' || progress.progress_percentage >= 100) {
              clearInterval(progressInterval);
            }
          }
        } catch (error) {
          console.warn('Failed to get progress update:', error);
        }
      }, 2000);

      // Clean up interval after 5 minutes max
      setTimeout(() => clearInterval(progressInterval), 5 * 60 * 1000);

      return result;
    } finally {
      if (onProgress) {
        this.removeProgressCallback(onProgress);
      }
    }
  }

  /**
   * Get current workflow progress
   */
  async getWorkflowProgress(): Promise<WorkflowProgress | null> {
    try {
      const response = await apiService.request('/api/v1/workflow/progress');
      return response;
    } catch (error) {
      console.error('Failed to get workflow progress:', error);
      return null;
    }
  }

  /**
   * Resume interrupted workflow
   */
  async resumeWorkflow(): Promise<WorkflowResult> {
    try {
      const response = await apiService.request('/api/v1/workflow/resume', {
        method: 'POST'
      });
      
      this.currentProgress = response.progress;
      this.notifyProgressCallbacks(response.progress);
      
      return response;
    } catch (error) {
      console.error('Failed to resume workflow:', error);
      throw error;
    }
  }

  /**
   * Validate workflow integrity
   */
  async validateWorkflowIntegrity(): Promise<ValidationResult> {
    try {
      const response = await apiService.request('/api/v1/workflow/validate');
      return response;
    } catch (error) {
      console.error('Failed to validate workflow integrity:', error);
      throw error;
    }
  }

  /**
   * Get available workflow stages
   */
  async getWorkflowStages(): Promise<string[]> {
    try {
      const response = await apiService.request('/api/v1/workflow/stages');
      return response;
    } catch (error) {
      console.error('Failed to get workflow stages:', error);
      return [];
    }
  }

  /**
   * Test integration components
   */
  async testIntegrationComponents(): Promise<Record<string, any>> {
    try {
      const response = await apiService.request('/api/v1/workflow/test-integration', {
        method: 'POST'
      });
      return response;
    } catch (error) {
      console.error('Failed to test integration components:', error);
      throw error;
    }
  }

  /**
   * Reset workflow progress
   */
  async resetWorkflow(): Promise<{ message: string; user_id: string }> {
    try {
      const response = await apiService.request('/api/v1/workflow/reset', {
        method: 'DELETE'
      });
      
      this.currentProgress = null;
      this.notifyProgressCallbacks(null);
      
      return response;
    } catch (error) {
      console.error('Failed to reset workflow:', error);
      throw error;
    }
  }

  /**
   * Execute resume upload only
   */
  async executeResumeUpload(resumeFile: File): Promise<any> {
    try {
      return await apiService.uploadResume(resumeFile);
    } catch (error) {
      console.error('Resume upload failed:', error);
      throw error;
    }
  }

  /**
   * Execute platform connections only
   */
  async executePlatformConnections(
    platformAccounts: Record<string, Record<string, string>>
  ): Promise<any> {
    try {
      const results = {};
      
      for (const [platform, accountData] of Object.entries(platformAccounts)) {
        try {
          const result = await apiService.request('/api/v1/platforms/connect', {
            method: 'POST',
            body: JSON.stringify({
              platform,
              ...accountData
            })
          });
          results[platform] = result;
        } catch (error) {
          console.error(`Failed to connect ${platform}:`, error);
          results[platform] = { error: error.message };
        }
      }
      
      return results;
    } catch (error) {
      console.error('Platform connections failed:', error);
      throw error;
    }
  }

  /**
   * Execute AI analysis only
   */
  async executeAIAnalysis(): Promise<any> {
    try {
      return await apiService.request('/api/v1/ai-analysis/analyze', {
        method: 'POST'
      });
    } catch (error) {
      console.error('AI analysis failed:', error);
      throw error;
    }
  }

  /**
   * Get integrated dashboard data
   */
  async getIntegratedDashboardData(): Promise<any> {
    try {
      return await apiService.request('/api/v1/dashboard/data');
    } catch (error) {
      console.error('Failed to get dashboard data:', error);
      throw error;
    }
  }

  /**
   * Add progress callback
   */
  addProgressCallback(callback: (progress: WorkflowProgress | null) => void): void {
    this.progressCallbacks.push(callback);
  }

  /**
   * Remove progress callback
   */
  removeProgressCallback(callback: (progress: WorkflowProgress | null) => void): void {
    const index = this.progressCallbacks.indexOf(callback);
    if (index > -1) {
      this.progressCallbacks.splice(index, 1);
    }
  }

  /**
   * Notify all progress callbacks
   */
  private notifyProgressCallbacks(progress: WorkflowProgress | null): void {
    this.progressCallbacks.forEach(callback => {
      try {
        callback(progress);
      } catch (error) {
        console.error('Progress callback error:', error);
      }
    });
  }

  /**
   * Get current progress (cached)
   */
  getCurrentProgress(): WorkflowProgress | null {
    return this.currentProgress;
  }

  /**
   * Check if workflow is in progress
   */
  isWorkflowInProgress(): boolean {
    return this.currentProgress !== null && 
           this.currentProgress.current_stage !== 'completed' && 
           this.currentProgress.progress_percentage < 100;
  }

  /**
   * Get workflow stage display name
   */
  getStageDisplayName(stage: string): string {
    const stageNames: Record<string, string> = {
      'profile_creation': 'Creating Profile',
      'resume_processing': 'Processing Resume',
      'platform_connection': 'Connecting Platforms',
      'data_scraping': 'Collecting Data',
      'ai_analysis': 'AI Analysis',
      'dashboard_preparation': 'Preparing Dashboard',
      'job_matching': 'Matching Jobs',
      'completed': 'Completed'
    };
    
    return stageNames[stage] || stage.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase());
  }

  /**
   * Get workflow stage icon
   */
  getStageIcon(stage: string): string {
    const stageIcons: Record<string, string> = {
      'profile_creation': '👤',
      'resume_processing': '📄',
      'platform_connection': '🔗',
      'data_scraping': '🔍',
      'ai_analysis': '🤖',
      'dashboard_preparation': '📊',
      'job_matching': '💼',
      'completed': '✅'
    };
    
    return stageIcons[stage] || '⚙️';
  }
}

export const workflowService = new WorkflowService();