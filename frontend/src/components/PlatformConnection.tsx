import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Badge } from './ui/badge';
import { Progress } from './ui/progress';
import { Separator } from './ui/separator';
import { toast } from 'sonner';
import { apiService } from '../services/api';
import {
  Github,
  Code,
  Linkedin,
  Trophy,
  Award,
  Database,
  CheckCircle,
  XCircle,
  AlertCircle,
  Clock,
  RefreshCw,
  Eye,
  EyeOff,
  Link as LinkIcon,
  Plus,
  Trash2,
  Settings
} from 'lucide-react';

// Platform configuration
type PlatformConfig = {
  name: string;
  icon: React.ComponentType<{ className?: string }>;
  color: string;
  placeholder: string;
  urlPattern: string;
  description: string;
  fields: Array<{
    key: string;
    label: string;
    type: string;
    required: boolean;
  }>;
};

const PLATFORMS: Record<string, PlatformConfig> = {
  github: {
    name: 'GitHub',
    icon: Github,
    color: 'bg-gray-900',
    placeholder: 'Enter GitHub username',
    urlPattern: 'https://github.com/{username}',
    description: 'Connect your GitHub to analyze repositories, contributions, and coding activity',
    fields: [
      { key: 'username', label: 'Username', type: 'text', required: true }
    ]
  },
  leetcode: {
    name: 'LeetCode',
    icon: Code,
    color: 'bg-orange-500',
    placeholder: 'Enter LeetCode username',
    urlPattern: 'https://leetcode.com/{username}',
    description: 'Connect LeetCode to track problem-solving statistics and contest ratings',
    fields: [
      { key: 'username', label: 'Username', type: 'text', required: true }
    ]
  },
  linkedin: {
    name: 'LinkedIn',
    icon: Linkedin,
    color: 'bg-blue-600',
    placeholder: 'Enter LinkedIn profile URL',
    urlPattern: '{url}',
    description: 'Connect LinkedIn to analyze professional experience and network',
    fields: [
      { key: 'url', label: 'Profile URL', type: 'url', required: true }
    ]
  },
  codeforces: {
    name: 'Codeforces',
    icon: Trophy,
    color: 'bg-red-500',
    placeholder: 'Enter Codeforces handle',
    urlPattern: 'https://codeforces.com/profile/{username}',
    description: 'Connect Codeforces for competitive programming statistics',
    fields: [
      { key: 'username', label: 'Handle', type: 'text', required: true }
    ]
  },
  atcoder: {
    name: 'AtCoder',
    icon: Award,
    color: 'bg-green-600',
    placeholder: 'Enter AtCoder username',
    urlPattern: 'https://atcoder.jp/users/{username}',
    description: 'Connect AtCoder for contest ratings and achievements',
    fields: [
      { key: 'username', label: 'Username', type: 'text', required: true }
    ]
  },
  hackerrank: {
    name: 'HackerRank',
    icon: Code,
    color: 'bg-green-500',
    placeholder: 'Enter HackerRank username',
    urlPattern: 'https://www.hackerrank.com/{username}',
    description: 'Connect HackerRank for skill certifications and challenges',
    fields: [
      { key: 'username', label: 'Username', type: 'text', required: true }
    ]
  },
  kaggle: {
    name: 'Kaggle',
    icon: Database,
    color: 'bg-blue-500',
    placeholder: 'Enter Kaggle username',
    urlPattern: 'https://www.kaggle.com/{username}',
    description: 'Connect Kaggle for competition rankings and dataset contributions',
    fields: [
      { key: 'username', label: 'Username', type: 'text', required: true }
    ]
  }
};

interface PlatformAccount {
  id: string;
  platform: string;
  username: string;
  profile_url?: string;
  is_active: boolean;
  is_verified: boolean;
  scraping_status: 'pending' | 'in_progress' | 'completed' | 'failed' | 'rate_limited' | 'unauthorized' | 'not_found';
  last_scraped_at?: string;
  data_completeness_score?: number;
  data_freshness_score?: number;
  statistics?: any;
  processed_data?: any;
  last_error?: string;
}

interface PlatformConnectionProps {
  onConnectionUpdate?: (accounts: PlatformAccount[]) => void;
  onCancel?: () => void;
}

export function PlatformConnection({ onConnectionUpdate, onCancel }: PlatformConnectionProps) {
  const [connectedAccounts, setConnectedAccounts] = useState<PlatformAccount[]>([]);
  const [loading, setLoading] = useState(true);
  const [connecting, setConnecting] = useState<string | null>(null);
  const [validating, setValidating] = useState<string | null>(null);
  const [showAddForm, setShowAddForm] = useState(false);
  const [selectedPlatform, setSelectedPlatform] = useState<string>('');
  const [formData, setFormData] = useState<Record<string, string>>({});
  const [validationResults, setValidationResults] = useState<Record<string, boolean>>({});
  const [previewData, setPreviewData] = useState<Record<string, any>>({});
  const [showPreview, setShowPreview] = useState<Record<string, boolean>>({});

  useEffect(() => {
    fetchConnectedAccounts();
  }, []);

  const fetchConnectedAccounts = async () => {
    try {
      setLoading(true);
      // This would be replaced with actual API call
      // const accounts = await apiService.getPlatformAccounts();
      
      // Mock data for demonstration
      const mockAccounts: PlatformAccount[] = [
        {
          id: '1',
          platform: 'github',
          username: 'johndoe',
          profile_url: 'https://github.com/johndoe',
          is_active: true,
          is_verified: true,
          scraping_status: 'completed',
          last_scraped_at: '2024-01-15T10:30:00Z',
          data_completeness_score: 0.95,
          data_freshness_score: 0.88,
          statistics: { repositories: 42, contributions: 1250, followers: 89 }
        },
        {
          id: '2',
          platform: 'leetcode',
          username: 'johndoe',
          is_active: true,
          is_verified: false,
          scraping_status: 'pending',
          data_completeness_score: 0.0,
          data_freshness_score: 0.0
        }
      ];
      
      setConnectedAccounts(mockAccounts);
    } catch (error) {
      toast.error('Failed to fetch connected accounts');
      console.error('Error fetching accounts:', error);
    } finally {
      setLoading(false);
    }
  };

  const handlePlatformSelect = (platform: string) => {
    setSelectedPlatform(platform);
    setFormData({});
    setValidationResults({});
    setShowAddForm(true);
  };

  const handleInputChange = (field: string, value: string) => {
    setFormData(prev => ({ ...prev, [field]: value }));
    // Clear validation when user types
    if (validationResults[field] !== undefined) {
      const newResults = { ...validationResults };
      delete newResults[field];
      setValidationResults(newResults);
    }
  };

  const validatePlatformAccount = async (platform: string, data: Record<string, string>) => {
    try {
      setValidating(platform);
      
      // Prepare validation request based on platform
      const validationRequest: any = {};
      
      if (platform === 'github') {
        validationRequest.github_username = data.username;
      } else if (platform === 'leetcode') {
        validationRequest.leetcode_username = data.username;
      } else if (platform === 'linkedin') {
        validationRequest.linkedin_url = data.url;
      }
      
      // Call validation API
      const response = await apiService.request('/api/v1/external-profiles/validate', {
        method: 'POST',
        body: JSON.stringify(validationRequest)
      });
      
      const isValid = response.validation_results[Object.keys(response.validation_results)[0]];
      setValidationResults(prev => ({ ...prev, [platform]: isValid }));
      
      if (isValid) {
        toast.success(`${PLATFORMS[platform].name} account validated successfully!`);
      } else {
        toast.error(`${PLATFORMS[platform].name} account validation failed`);
      }
      
      return isValid;
    } catch (error) {
      setValidationResults(prev => ({ ...prev, [platform]: false }));
      toast.error(`Failed to validate ${PLATFORMS[platform].name} account`);
      return false;
    } finally {
      setValidating(null);
    }
  };

  const connectPlatformAccount = async () => {
    if (!selectedPlatform || !formData) return;
    
    const platform = PLATFORMS[selectedPlatform];
    const isValid = await validatePlatformAccount(selectedPlatform, formData);
    
    if (!isValid) return;
    
    try {
      setConnecting(selectedPlatform);
      
      // Create platform account
      const accountData = {
        platform: selectedPlatform,
        username: formData.username || formData.url?.split('/').pop(),
        profile_url: platform.urlPattern.replace('{username}', formData.username || '').replace('{url}', formData.url || ''),
        ...formData
      };
      
      // This would be replaced with actual API call
      // const newAccount = await apiService.createPlatformAccount(accountData);
      
      // Mock response
      const newAccount: PlatformAccount = {
        id: Date.now().toString(),
        platform: selectedPlatform,
        username: formData.username || formData.url?.split('/').pop() || '',
        profile_url: accountData.profile_url,
        is_active: true,
        is_verified: false,
        scraping_status: 'pending',
        data_completeness_score: 0.0,
        data_freshness_score: 0.0
      };
      
      setConnectedAccounts(prev => [...prev, newAccount]);
      setShowAddForm(false);
      setSelectedPlatform('');
      setFormData({});
      
      toast.success(`${platform.name} account connected successfully!`);
      
      // Start data collection
      setTimeout(() => {
        startDataCollection(newAccount.id);
      }, 1000);
      
    } catch (error) {
      toast.error(`Failed to connect ${platform.name} account`);
      console.error('Error connecting account:', error);
    } finally {
      setConnecting(null);
    }
  };

  const startDataCollection = async (accountId: string) => {
    try {
      setConnectedAccounts(prev => 
        prev.map(account => 
          account.id === accountId 
            ? { ...account, scraping_status: 'in_progress' }
            : account
        )
      );
      
      // Simulate data collection progress
      let progress = 0;
      const interval = setInterval(() => {
        progress += Math.random() * 20;
        if (progress >= 100) {
          clearInterval(interval);
          // Mark as completed
          setConnectedAccounts(prev => 
            prev.map(account => 
              account.id === accountId 
                ? { 
                    ...account, 
                    scraping_status: 'completed',
                    is_verified: true,
                    data_completeness_score: 0.85 + Math.random() * 0.15,
                    data_freshness_score: 0.90 + Math.random() * 0.10,
                    last_scraped_at: new Date().toISOString(),
                    statistics: generateMockStatistics(account.platform)
                  }
                : account
            )
          );
          toast.success('Data collection completed!');
        }
      }, 1500);
      
    } catch (error) {
      setConnectedAccounts(prev => 
        prev.map(account => 
          account.id === accountId 
            ? { ...account, scraping_status: 'failed', last_error: 'Data collection failed' }
            : account
        )
      );
      toast.error('Data collection failed');
    }
  };

  const generateMockStatistics = (platform: string) => {
    switch (platform) {
      case 'github':
        return { repositories: Math.floor(Math.random() * 50) + 10, contributions: Math.floor(Math.random() * 1000) + 200, followers: Math.floor(Math.random() * 100) + 10 };
      case 'leetcode':
        return { problems_solved: Math.floor(Math.random() * 500) + 50, contest_rating: Math.floor(Math.random() * 1000) + 1200, acceptance_rate: (Math.random() * 30 + 70).toFixed(1) + '%' };
      case 'linkedin':
        return { connections: Math.floor(Math.random() * 500) + 100, endorsements: Math.floor(Math.random() * 50) + 10, posts: Math.floor(Math.random() * 20) + 5 };
      default:
        return { score: Math.floor(Math.random() * 1000) + 500, rank: Math.floor(Math.random() * 10000) + 1000 };
    }
  };

  const disconnectAccount = async (accountId: string) => {
    try {
      // This would be replaced with actual API call
      // await apiService.deletePlatformAccount(accountId);
      
      setConnectedAccounts(prev => prev.filter(account => account.id !== accountId));
      toast.success('Account disconnected successfully');
    } catch (error) {
      toast.error('Failed to disconnect account');
    }
  };

  const refreshAccountData = async (accountId: string) => {
    const account = connectedAccounts.find(acc => acc.id === accountId);
    if (!account) return;
    
    try {
      setConnectedAccounts(prev => 
        prev.map(acc => 
          acc.id === accountId 
            ? { ...acc, scraping_status: 'in_progress' }
            : acc
        )
      );
      
      // Simulate refresh
      setTimeout(() => {
        setConnectedAccounts(prev => 
          prev.map(acc => 
            acc.id === accountId 
              ? { 
                  ...acc, 
                  scraping_status: 'completed',
                  last_scraped_at: new Date().toISOString(),
                  data_freshness_score: 0.95 + Math.random() * 0.05,
                  statistics: generateMockStatistics(acc.platform)
                }
              : acc
          )
        );
        toast.success('Account data refreshed successfully');
      }, 2000);
      
    } catch (error) {
      toast.error('Failed to refresh account data');
    }
  };

  const togglePreview = (accountId: string) => {
    setShowPreview(prev => ({ ...prev, [accountId]: !prev[accountId] }));
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'in_progress':
        return <RefreshCw className="w-4 h-4 text-blue-500 animate-spin" />;
      case 'failed':
        return <XCircle className="w-4 h-4 text-red-500" />;
      case 'pending':
        return <Clock className="w-4 h-4 text-yellow-500" />;
      default:
        return <AlertCircle className="w-4 h-4 text-gray-500" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'bg-green-100 text-green-800';
      case 'in_progress':
        return 'bg-blue-100 text-blue-800';
      case 'failed':
        return 'bg-red-100 text-red-800';
      case 'pending':
        return 'bg-yellow-100 text-yellow-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center p-8">
        <RefreshCw className="w-6 h-6 animate-spin mr-2" />
        <span>Loading platform connections...</span>
      </div>
    );
  }

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
              <LinkIcon className="w-6 h-6" />
              <span>Platform Connections</span>
            </CardTitle>
            <p className="text-muted-foreground">
              Connect your accounts from various platforms to get comprehensive career insights
            </p>
          </CardHeader>
          <CardContent>
            <div className="flex justify-between items-center mb-6">
              <div className="text-sm text-muted-foreground">
                {connectedAccounts.length} platform{connectedAccounts.length !== 1 ? 's' : ''} connected
              </div>
              <Button onClick={() => setShowAddForm(true)} className="flex items-center space-x-2">
                <Plus className="w-4 h-4" />
                <span>Add Platform</span>
              </Button>
            </div>

            {/* Connected Accounts */}
            <div className="space-y-4">
              {connectedAccounts.map((account) => {
                const platform = PLATFORMS[account.platform];
                const IconComponent = platform.icon;
                
                return (
                  <motion.div
                    key={account.id}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.4 }}
                  >
                    <Card className="border-l-4" style={{ borderLeftColor: platform.color.replace('bg-', '#') }}>
                      <CardContent className="p-4">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center space-x-4">
                            <div className={`p-2 rounded-lg ${platform.color} text-white`}>
                              <IconComponent className="w-5 h-5" />
                            </div>
                            <div>
                              <h3 className="font-semibold">{platform.name}</h3>
                              <p className="text-sm text-muted-foreground">@{account.username}</p>
                              {account.profile_url && (
                                <a 
                                  href={account.profile_url} 
                                  target="_blank" 
                                  rel="noopener noreferrer"
                                  className="text-xs text-blue-600 hover:underline"
                                >
                                  View Profile
                                </a>
                              )}
                            </div>
                          </div>
                          
                          <div className="flex items-center space-x-3">
                            {/* Status Badge */}
                            <Badge className={`${getStatusColor(account.scraping_status)} flex items-center space-x-1`}>
                              {getStatusIcon(account.scraping_status)}
                              <span className="capitalize">{account.scraping_status.replace('_', ' ')}</span>
                            </Badge>
                            
                            {/* Verification Badge */}
                            {account.is_verified && (
                              <Badge variant="secondary" className="bg-green-100 text-green-800">
                                <CheckCircle className="w-3 h-3 mr-1" />
                                Verified
                              </Badge>
                            )}
                            
                            {/* Action Buttons */}
                            <div className="flex space-x-1">
                              <Button
                                variant="ghost"
                                size="sm"
                                onClick={() => togglePreview(account.id)}
                                title="Toggle preview"
                              >
                                {showPreview[account.id] ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                              </Button>
                              <Button
                                variant="ghost"
                                size="sm"
                                onClick={() => refreshAccountData(account.id)}
                                disabled={account.scraping_status === 'in_progress'}
                                title="Refresh data"
                              >
                                <RefreshCw className={`w-4 h-4 ${account.scraping_status === 'in_progress' ? 'animate-spin' : ''}`} />
                              </Button>
                              <Button
                                variant="ghost"
                                size="sm"
                                onClick={() => disconnectAccount(account.id)}
                                title="Disconnect account"
                              >
                                <Trash2 className="w-4 h-4 text-red-500" />
                              </Button>
                            </div>
                          </div>
                        </div>
                        
                        {/* Data Quality Indicators */}
                        {account.scraping_status === 'completed' && (
                          <div className="mt-4 grid grid-cols-2 gap-4">
                            <div>
                              <div className="flex justify-between text-sm mb-1">
                                <span>Data Completeness</span>
                                <span>{Math.round((account.data_completeness_score || 0) * 100)}%</span>
                              </div>
                              <Progress value={(account.data_completeness_score || 0) * 100} className="h-2" />
                            </div>
                            <div>
                              <div className="flex justify-between text-sm mb-1">
                                <span>Data Freshness</span>
                                <span>{Math.round((account.data_freshness_score || 0) * 100)}%</span>
                              </div>
                              <Progress value={(account.data_freshness_score || 0) * 100} className="h-2" />
                            </div>
                          </div>
                        )}
                        
                        {/* Last Updated */}
                        {account.last_scraped_at && (
                          <div className="mt-2 text-xs text-muted-foreground">
                            Last updated: {new Date(account.last_scraped_at).toLocaleString()}
                          </div>
                        )}
                        
                        {/* Error Message */}
                        {account.scraping_status === 'failed' && account.last_error && (
                          <div className="mt-2 p-2 bg-red-50 border border-red-200 rounded text-sm text-red-800">
                            Error: {account.last_error}
                          </div>
                        )}
                        
                        {/* Data Preview */}
                        <AnimatePresence>
                          {showPreview[account.id] && account.statistics && (
                            <motion.div
                              initial={{ opacity: 0, height: 0 }}
                              animate={{ opacity: 1, height: 'auto' }}
                              exit={{ opacity: 0, height: 0 }}
                              transition={{ duration: 0.3 }}
                              className="mt-4 p-3 bg-gray-50 rounded-lg"
                            >
                              <h4 className="font-medium mb-2">Platform Statistics</h4>
                              <div className="grid grid-cols-3 gap-4 text-sm">
                                {Object.entries(account.statistics).map(([key, value]) => (
                                  <div key={key} className="text-center">
                                    <div className="font-semibold">{String(value)}</div>
                                    <div className="text-muted-foreground capitalize">
                                      {key.replace('_', ' ')}
                                    </div>
                                  </div>
                                ))}
                              </div>
                            </motion.div>
                          )}
                        </AnimatePresence>
                      </CardContent>
                    </Card>
                  </motion.div>
                );
              })}
            </div>
            
            {connectedAccounts.length === 0 && (
              <div className="text-center py-8">
                <LinkIcon className="w-12 h-12 mx-auto text-muted-foreground mb-4" />
                <h3 className="text-lg font-semibold mb-2">No platforms connected</h3>
                <p className="text-muted-foreground mb-4">
                  Connect your accounts to get comprehensive career insights
                </p>
                <Button onClick={() => setShowAddForm(true)}>
                  <Plus className="w-4 h-4 mr-2" />
                  Add Your First Platform
                </Button>
              </div>
            )}
          </CardContent>
        </Card>
      </motion.div>

      {/* Add Platform Modal */}
      <AnimatePresence>
        {showAddForm && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4"
          >
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              className="bg-white rounded-lg max-w-4xl w-full max-h-[90vh] overflow-y-auto"
            >
              <div className="sticky top-0 bg-white border-b p-4 flex items-center justify-between">
                <h2 className="text-xl font-semibold">Add Platform Connection</h2>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => {
                    setShowAddForm(false);
                    setSelectedPlatform('');
                    setFormData({});
                  }}
                >
                  ×
                </Button>
              </div>
              
              <div className="p-6">
                {!selectedPlatform ? (
                  <div>
                    <h3 className="text-lg font-semibold mb-4">Choose a platform to connect</h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                      {Object.entries(PLATFORMS).map(([key, platform]) => {
                        const IconComponent = platform.icon;
                        const isConnected = connectedAccounts.some(acc => acc.platform === key);
                        
                        return (
                          <Card
                            key={key}
                            className={`cursor-pointer transition-all hover:shadow-md ${
                              isConnected ? 'opacity-50' : 'hover:border-primary'
                            }`}
                            onClick={() => !isConnected && handlePlatformSelect(key)}
                          >
                            <CardContent className="p-4">
                              <div className="flex items-center space-x-3 mb-3">
                                <div className={`p-2 rounded-lg ${platform.color} text-white`}>
                                  <IconComponent className="w-5 h-5" />
                                </div>
                                <div>
                                  <h4 className="font-semibold">{platform.name}</h4>
                                  {isConnected && (
                                    <Badge variant="secondary" className="text-xs">
                                      Already Connected
                                    </Badge>
                                  )}
                                </div>
                              </div>
                              <p className="text-sm text-muted-foreground">
                                {platform.description}
                              </p>
                            </CardContent>
                          </Card>
                        );
                      })}
                    </div>
                  </div>
                ) : (
                  <div>
                    <div className="flex items-center space-x-3 mb-6">
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => setSelectedPlatform('')}
                      >
                        ← Back
                      </Button>
                      <div className={`p-2 rounded-lg ${PLATFORMS[selectedPlatform].color} text-white`}>
                        {React.createElement(PLATFORMS[selectedPlatform].icon, { className: "w-5 h-5" })}
                      </div>
                      <div>
                        <h3 className="text-lg font-semibold">Connect {PLATFORMS[selectedPlatform].name}</h3>
                        <p className="text-sm text-muted-foreground">
                          {PLATFORMS[selectedPlatform].description}
                        </p>
                      </div>
                    </div>
                    
                    <div className="space-y-4">
                      {PLATFORMS[selectedPlatform].fields.map((field) => (
                        <div key={field.key}>
                          <Label htmlFor={field.key}>{field.label}</Label>
                          <Input
                            id={field.key}
                            type={field.type}
                            placeholder={PLATFORMS[selectedPlatform].placeholder}
                            value={formData[field.key] || ''}
                            onChange={(e) => handleInputChange(field.key, e.target.value)}
                            className={
                              validationResults[selectedPlatform] === false
                                ? 'border-red-500'
                                : validationResults[selectedPlatform] === true
                                ? 'border-green-500'
                                : ''
                            }
                          />
                          {validationResults[selectedPlatform] === true && (
                            <div className="flex items-center space-x-1 mt-1 text-sm text-green-600">
                              <CheckCircle className="w-4 h-4" />
                              <span>Account validated successfully</span>
                            </div>
                          )}
                          {validationResults[selectedPlatform] === false && (
                            <div className="flex items-center space-x-1 mt-1 text-sm text-red-600">
                              <XCircle className="w-4 h-4" />
                              <span>Account validation failed</span>
                            </div>
                          )}
                        </div>
                      ))}
                      
                      <div className="flex space-x-3 pt-4">
                        <Button
                          onClick={connectPlatformAccount}
                          disabled={
                            connecting === selectedPlatform ||
                            validating === selectedPlatform ||
                            !Object.values(formData).some(value => value.trim())
                          }
                          className="flex-1"
                        >
                          {connecting === selectedPlatform ? (
                            <>
                              <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                              Connecting...
                            </>
                          ) : validating === selectedPlatform ? (
                            <>
                              <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                              Validating...
                            </>
                          ) : (
                            <>
                              <Plus className="w-4 h-4 mr-2" />
                              Connect Account
                            </>
                          )}
                        </Button>
                        <Button
                          variant="outline"
                          onClick={() => {
                            setShowAddForm(false);
                            setSelectedPlatform('');
                            setFormData({});
                          }}
                        >
                          Cancel
                        </Button>
                      </div>
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
}