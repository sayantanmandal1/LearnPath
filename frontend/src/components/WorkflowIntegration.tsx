/**
 * Workflow Integration Component
 * 
 * This component orchestrates the complete user workflow, providing a unified interface
 * for resume upload, platform connections, AI analysis, and dashboard integration.
 */

import React, { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Progress } from './ui/progress';
import { Badge } from './ui/badge';
import { Alert, AlertDescription } from './ui/alert';
import { Separator } from './ui/separator';
import { toast } from 'sonner';
import { 
  workflowService, 
  WorkflowProgress, 
  WorkflowResult,
  ValidationResult 
} from '../services/workflowService';
import { ResumeUpload } from './ResumeUpload';
import { PlatformConnection } from './PlatformConnection';
import { EnhancedDashboard } from './EnhancedDashboard';
import {
  CheckCircle,
  XCircle,
  Clock,
  AlertCircle,
  RefreshCw,
  Play,
  Pause,
  RotateCcw,
  Eye,
  Settings,
  Zap,
  TrendingUp,
  Users,
  Brain,
  FileText,
  Link as LinkIcon,
  BarChart3
} from 'lucide-react';

interface WorkflowIntegrationProps {
  onComplete?: (result: WorkflowResult) => void;
  onError?: (error: string) => void;
  autoStart?: boolean;
  showProgress?: boolean;
}

type WorkflowStep = 'upload' | 'platforms' | 'analysis' | 'dashboard' | 'complete';

export function WorkflowIntegration({ 
  onComplete, 
  onError, 
  autoStart = false,
  showProgress = true 
}: WorkflowIntegrationProps) {
  const [currentStep, setCurrentStep] = useState<WorkflowStep>('upload');
  const [workflowProgress, setWorkflowProgress] = useState<WorkflowProgress | null>(null);
  const [workflowResult, setWorkflowResult] = useState<WorkflowResult | null>(null);
  const [isExecuting, setIsExecuting] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [errors, setErrors] = useState<string[]>([]);
  const [warnings, setWarnings] = useState<string[]>([]);
  const [validationResult, setValidationResult] = useState<ValidationResult | null>(null);
  
  // Workflow data
  const [resumeFile, setResumeFile] = useState<File | null>(null);
  const [resumeData, setResumeData] = useState<any>(null);
  const [platformAccounts, setPlatformAccounts] = useState<Record<string, any>>({});
  const [analysisResults, setAnalysisResults] = useState<any>(null);
  const [dashboardData, setDashboardData] = useState<any>(null);

  // Progress tracking
  useEffect(() => {
    const handleProgress = (progress: WorkflowProgress | null) => {
      setWorkflowProgress(progress);
      if (progress) {
        setErrors(progress.errors);
        setWarnings(progress.warnings);
      }
    };

    workflowService.addProgressCallback(handleProgress);
    
    // Check for existing progress on mount
    const checkExistingProgress = async () => {
      try {
        const progress = await workflowService.getWorkflowProgress();
        if (progress) {
          setWorkflowProgress(progress);
          setIsExecuting(progress.current_stage !== 'completed' && progress.progress_percentage < 100);
        }
      } catch (error) {
        console.error('Failed to check existing progress:', error);
      }
    };

    checkExistingProgress();

    return () => {
      workflowService.removeProgressCallback(handleProgress);
    };
  }, []);

  // Auto-start workflow if requested
  useEffect(() => {
    if (autoStart && !isExecuting && !workflowProgress) {
      handleStartWorkflow();
    }
  }, [autoStart]);

  const handleStartWorkflow = async () => {
    try {
      setIsExecuting(true);
      setErrors([]);
      setWarnings([]);
      
      toast.info('Starting complete workflow...');
      
      const result = await workflowService.executeWorkflowWithProgress(
        resumeFile || undefined,
        Object.keys(platformAccounts).length > 0 ? platformAccounts : undefined,
        (progress) => {
          setWorkflowProgress(progress);
          if (progress) {
            // Update current step based on progress
            updateCurrentStepFromProgress(progress);
          }
        }
      );
      
      setWorkflowResult(result);
      
      if (result.success) {
        toast.success('Workflow completed successfully!');
        setCurrentStep('complete');
        onComplete?.(result);
      } else {
        toast.error('Workflow completed with errors');
        setErrors(result.errors);
        onError?.(result.errors.join(', '));
      }
      
    } catch (error) {
      console.error('Workflow execution failed:', error);
      toast.error('Workflow execution failed');
      setErrors([error instanceof Error ? error.message : 'Unknown error']);
      onError?.(error instanceof Error ? error.message : 'Unknown error');
    } finally {
      setIsExecuting(false);
    }
  };

  const handleResumeWorkflow = async () => {
    try {
      setIsExecuting(true);
      toast.info('Resuming workflow...');
      
      const result = await workflowService.resumeWorkflow();
      setWorkflowResult(result);
      
      if (result.success) {
        toast.success('Workflow resumed successfully!');
      } else {
        toast.error('Failed to resume workflow');
        setErrors(result.errors);
      }
      
    } catch (error) {
      console.error('Failed to resume workflow:', error);
      toast.error('Failed to resume workflow');
      setErrors([error instanceof Error ? error.message : 'Unknown error']);
    } finally {
      setIsExecuting(false);
    }
  };

  const handleResetWorkflow = async () => {
    try {
      await workflowService.resetWorkflow();
      setWorkflowProgress(null);
      setWorkflowResult(null);
      setCurrentStep('upload');
      setErrors([]);
      setWarnings([]);
      setResumeFile(null);
      setResumeData(null);
      setPlatformAccounts({});
      setAnalysisResults(null);
      setDashboardData(null);
      toast.success('Workflow reset successfully');
    } catch (error) {
      console.error('Failed to reset workflow:', error);
      toast.error('Failed to reset workflow');
    }
  };

  const handleValidateIntegrity = async () => {
    try {
      const result = await workflowService.validateWorkflowIntegrity();
      setValidationResult(result);
      
      if (result.overall_status === 'passed') {
        toast.success('Workflow integrity validated successfully');
      } else {
        toast.warning(`Validation completed with status: ${result.overall_status}`);
      }
    } catch (error) {
      console.error('Failed to validate workflow integrity:', error);
      toast.error('Failed to validate workflow integrity');
    }
  };

  const updateCurrentStepFromProgress = (progress: WorkflowProgress) => {
    const stageToStep: Record<string, WorkflowStep> = {
      'profile_creation': 'upload',
      'resume_processing': 'upload',
      'platform_connection': 'platforms',
      'data_scraping': 'platforms',
      'ai_analysis': 'analysis',
      'dashboard_preparation': 'dashboard',
      'job_matching': 'dashboard',
      'completed': 'complete'
    };
    
    const step = stageToStep[progress.current_stage];
    if (step) {
      setCurrentStep(step);
    }
  };

  const handleResumeUploadComplete = (data: any) => {
    setResumeData(data);
    toast.success('Resume processed successfully!');
    
    if (!isExecuting) {
      setCurrentStep('platforms');
    }
  };

  const handlePlatformConnectionUpdate = (accounts: any[]) => {
    const accountsMap = accounts.reduce((acc, account) => {
      acc[account.platform] = {
        username: account.username,
        profile_url: account.profile_url
      };
      return acc;
    }, {});
    
    setPlatformAccounts(accountsMap);
    toast.success('Platform connections updated!');
    
    if (!isExecuting && Object.keys(accountsMap).length > 0) {
      setCurrentStep('analysis');
    }
  };

  const getStageIcon = (stage: string) => {
    const icons: Record<string, React.ReactNode> = {
      'profile_creation': <Users className="w-4 h-4" />,
      'resume_processing': <FileText className="w-4 h-4" />,
      'platform_connection': <LinkIcon className="w-4 h-4" />,
      'data_scraping': <RefreshCw className="w-4 h-4" />,
      'ai_analysis': <Brain className="w-4 h-4" />,
      'dashboard_preparation': <BarChart3 className="w-4 h-4" />,
      'job_matching': <TrendingUp className="w-4 h-4" />,
      'completed': <CheckCircle className="w-4 h-4" />
    };
    
    return icons[stage] || <Settings className="w-4 h-4" />;
  };

  const getStageStatus = (stage: string) => {
    if (!workflowProgress) return 'pending';
    
    if (workflowProgress.completed_stages.includes(stage)) {
      return 'completed';
    } else if (workflowProgress.current_stage === stage) {
      return 'in_progress';
    } else {
      return 'pending';
    }
  };

  const renderProgressSection = () => {
    if (!showProgress || !workflowProgress) return null;

    return (
      <Card className="mb-6">
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Zap className="w-5 h-5" />
            <span>Workflow Progress</span>
            <Badge variant="outline">
              {workflowProgress.progress_percentage.toFixed(0)}%
            </Badge>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <Progress value={workflowProgress.progress_percentage} className="w-full" />
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {[
              'profile_creation',
              'resume_processing', 
              'platform_connection',
              'data_scraping',
              'ai_analysis',
              'dashboard_preparation',
              'job_matching',
              'completed'
            ].map((stage) => {
              const status = getStageStatus(stage);
              const isActive = workflowProgress.current_stage === stage;
              
              return (
                <div
                  key={stage}
                  className={`flex items-center space-x-2 p-2 rounded-lg ${
                    isActive ? 'bg-primary/10 border border-primary/20' : 'bg-muted/50'
                  }`}
                >
                  <div className={`p-1 rounded ${
                    status === 'completed' ? 'bg-green-100 text-green-600' :
                    status === 'in_progress' ? 'bg-blue-100 text-blue-600' :
                    'bg-gray-100 text-gray-400'
                  }`}>
                    {status === 'in_progress' ? (
                      <RefreshCw className="w-3 h-3 animate-spin" />
                    ) : status === 'completed' ? (
                      <CheckCircle className="w-3 h-3" />
                    ) : (
                      getStageIcon(stage)
                    )}
                  </div>
                  <span className="text-xs font-medium">
                    {workflowService.getStageDisplayName(stage)}
                  </span>
                </div>
              );
            })}
          </div>
          
          {(errors.length > 0 || warnings.length > 0) && (
            <div className="space-y-2">
              {errors.map((error, index) => (
                <Alert key={`error-${index}`} variant="destructive">
                  <XCircle className="h-4 w-4" />
                  <AlertDescription>{error}</AlertDescription>
                </Alert>
              ))}
              {warnings.map((warning, index) => (
                <Alert key={`warning-${index}`}>
                  <AlertCircle className="h-4 w-4" />
                  <AlertDescription>{warning}</AlertDescription>
                </Alert>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    );
  };

  const renderWorkflowControls = () => (
    <Card className="mb-6">
      <CardHeader>
        <CardTitle>Workflow Controls</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="flex flex-wrap gap-2">
          <Button
            onClick={handleStartWorkflow}
            disabled={isExecuting}
            className="flex items-center space-x-2"
          >
            <Play className="w-4 h-4" />
            <span>Start Complete Workflow</span>
          </Button>
          
          {workflowProgress && workflowProgress.current_stage !== 'completed' && (
            <Button
              onClick={handleResumeWorkflow}
              disabled={isExecuting}
              variant="outline"
              className="flex items-center space-x-2"
            >
              <RefreshCw className="w-4 h-4" />
              <span>Resume Workflow</span>
            </Button>
          )}
          
          <Button
            onClick={handleResetWorkflow}
            disabled={isExecuting}
            variant="outline"
            className="flex items-center space-x-2"
          >
            <RotateCcw className="w-4 h-4" />
            <span>Reset</span>
          </Button>
          
          <Button
            onClick={handleValidateIntegrity}
            variant="outline"
            className="flex items-center space-x-2"
          >
            <Eye className="w-4 h-4" />
            <span>Validate</span>
          </Button>
        </div>
        
        {validationResult && (
          <div className="mt-4 p-4 border rounded-lg">
            <h4 className="font-medium mb-2">Validation Results</h4>
            <Badge variant={
              validationResult.overall_status === 'passed' ? 'default' :
              validationResult.overall_status === 'warning' ? 'secondary' : 'destructive'
            }>
              {validationResult.overall_status}
            </Badge>
            {validationResult.issues.length > 0 && (
              <ul className="mt-2 text-sm text-muted-foreground">
                {validationResult.issues.map((issue, index) => (
                  <li key={index}>• {issue}</li>
                ))}
              </ul>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );

  const renderStepContent = () => {
    switch (currentStep) {
      case 'upload':
        return (
          <ResumeUpload
            onUploadComplete={handleResumeUploadComplete}
            onCancel={() => setCurrentStep('platforms')}
          />
        );
      
      case 'platforms':
        return (
          <PlatformConnection
            onConnectionUpdate={handlePlatformConnectionUpdate}
            onCancel={() => setCurrentStep('analysis')}
          />
        );
      
      case 'analysis':
        return (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Brain className="w-5 h-5" />
                <span>AI Analysis</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-muted-foreground mb-4">
                AI analysis will be performed automatically as part of the complete workflow.
              </p>
              <Button onClick={() => setCurrentStep('dashboard')}>
                Continue to Dashboard
              </Button>
            </CardContent>
          </Card>
        );
      
      case 'dashboard':
        return <EnhancedDashboard />;
      
      case 'complete':
        return (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <CheckCircle className="w-5 h-5 text-green-600" />
                <span>Workflow Complete!</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-muted-foreground mb-4">
                Your complete career analysis workflow has been successfully completed.
                All your data has been processed and integrated.
              </p>
              
              {workflowResult && (
                <div className="space-y-4">
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="text-center">
                      <div className="text-2xl font-bold text-green-600">
                        {workflowResult.execution_time.toFixed(1)}s
                      </div>
                      <div className="text-sm text-muted-foreground">Execution Time</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-blue-600">
                        {workflowResult.resume_data ? '✓' : '✗'}
                      </div>
                      <div className="text-sm text-muted-foreground">Resume Processed</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-purple-600">
                        {workflowResult.platform_data ? Object.keys(workflowResult.platform_data).length : 0}
                      </div>
                      <div className="text-sm text-muted-foreground">Platforms Connected</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-orange-600">
                        {workflowResult.job_matches ? workflowResult.job_matches.length : 0}
                      </div>
                      <div className="text-sm text-muted-foreground">Job Matches</div>
                    </div>
                  </div>
                  
                  <Button onClick={() => setCurrentStep('dashboard')} className="w-full">
                    View Your Dashboard
                  </Button>
                </div>
              )}
            </CardContent>
          </Card>
        );
      
      default:
        return null;
    }
  };

  return (
    <div className="max-w-6xl mx-auto p-6 space-y-6">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Zap className="w-6 h-6" />
              <span>Complete Workflow Integration</span>
            </CardTitle>
            <p className="text-muted-foreground">
              Streamlined process to upload your resume, connect platforms, analyze your profile, 
              and get personalized career recommendations.
            </p>
          </CardHeader>
        </Card>
      </motion.div>

      {renderProgressSection()}
      {renderWorkflowControls()}

      <AnimatePresence mode="wait">
        <motion.div
          key={currentStep}
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: -20 }}
          transition={{ duration: 0.5 }}
        >
          {renderStepContent()}
        </motion.div>
      </AnimatePresence>
    </div>
  );
}