/**
 * Learning Paths Card with personalized progress tracking
 */
import React, { useState } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card';
import { Badge } from '../ui/badge';
import { Button } from '../ui/button';
import { Progress } from '../ui/progress';
import { 
  BookOpen, 
  Clock, 
  Target, 
  TrendingUp, 
  CheckCircle, 
  PlayCircle,
  Star,
  Award,
  RefreshCw,
  ChevronRight,
  Calendar,
  Users
} from 'lucide-react';

interface LearningModule {
  id: string;
  title: string;
  description: string;
  duration: string;
  completed: boolean;
  progress: number;
  type: 'video' | 'article' | 'exercise' | 'project';
}

interface LearningPath {
  id: string;
  title: string;
  description: string;
  target_skills: string[];
  learning_modules: LearningModule[];
  estimated_duration: string;
  difficulty_level: 'Beginner' | 'Intermediate' | 'Advanced';
  progress: number;
  match_score: number;
  resources: {
    courses: string[];
    articles: string[];
    projects: string[];
  };
  completion_benefits: string[];
  career_impact: string;
}

interface LearningPathsCardProps {
  learningPaths: LearningPath[];
  loading?: boolean;
  onRefresh?: () => void;
  onPathStart?: (path: LearningPath) => void;
  onPathContinue?: (path: LearningPath) => void;
  className?: string;
}

export const LearningPathsCard: React.FC<LearningPathsCardProps> = ({
  learningPaths = [],
  loading = false,
  onRefresh,
  onPathStart,
  onPathContinue,
  className = '',
}) => {
  const [selectedPath, setSelectedPath] = useState<LearningPath | null>(null);
  const [showDetails, setShowDetails] = useState(false);

  const handlePathSelect = (path: LearningPath) => {
    setSelectedPath(path);
    setShowDetails(true);
  };

  const getDifficultyColor = (level: string) => {
    switch (level.toLowerCase()) {
      case 'beginner':
        return 'text-green-600 bg-green-50 border-green-200';
      case 'intermediate':
        return 'text-blue-600 bg-blue-50 border-blue-200';
      case 'advanced':
        return 'text-purple-600 bg-purple-50 border-purple-200';
      default:
        return 'text-gray-600 bg-gray-50 border-gray-200';
    }
  };

  const getMatchColor = (score: number) => {
    if (score >= 0.8) return 'text-green-600';
    if (score >= 0.6) return 'text-blue-600';
    return 'text-orange-600';
  };

  const formatDuration = (duration: string) => {
    if (!duration) return 'Self-paced';
    return duration.replace(/(\d+)\s*weeks?/i, '$1w').replace(/(\d+)\s*months?/i, '$1m');
  };

  const getProgressStatus = (progress: number) => {
    if (progress === 0) return { status: 'Not Started', color: 'text-gray-500' };
    if (progress < 100) return { status: 'In Progress', color: 'text-blue-600' };
    return { status: 'Completed', color: 'text-green-600' };
  };

  const getModuleIcon = (type: string) => {
    switch (type) {
      case 'video':
        return PlayCircle;
      case 'article':
        return BookOpen;
      case 'exercise':
        return Target;
      case 'project':
        return Award;
      default:
        return BookOpen;
    }
  };

  return (
    <Card className={`h-full ${className}`}>
      <CardHeader className="pb-4">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center space-x-2">
            <BookOpen className="w-5 h-5 text-blue-600" />
            <span>Learning Paths</span>
            <Badge variant="secondary" className="ml-2">
              {learningPaths.length} paths
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
        
        <div className="flex items-center space-x-4 text-sm text-muted-foreground mt-2">
          <div className="flex items-center space-x-1">
            <Target className="w-4 h-4" />
            <span>Personalized for you</span>
          </div>
          <div className="flex items-center space-x-1">
            <TrendingUp className="w-4 h-4" />
            <span>Market-aligned</span>
          </div>
        </div>
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
        ) : learningPaths.length === 0 ? (
          <div className="text-center py-8 text-muted-foreground">
            <BookOpen className="w-12 h-12 mx-auto mb-4 opacity-50" />
            <p>No learning paths available</p>
            <p className="text-sm">Complete your profile to get personalized recommendations</p>
          </div>
        ) : (
          <div className="space-y-4 max-h-96 overflow-y-auto">
            <AnimatePresence>
              {learningPaths.map((path, index) => {
                const progressStatus = getProgressStatus(path.progress);
                
                return (
                  <motion.div
                    key={path.id}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    transition={{ duration: 0.3, delay: index * 0.1 }}
                    className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow cursor-pointer"
                    onClick={() => handlePathSelect(path)}
                  >
                    <div className="flex items-start justify-between mb-3">
                      <div className="flex-1">
                        <h4 className="font-semibold text-gray-900 mb-1">{path.title}</h4>
                        <p className="text-sm text-muted-foreground mb-2 line-clamp-2">
                          {path.description}
                        </p>
                      </div>
                      <div className="flex flex-col items-end space-y-1">
                        <Badge className={`${getDifficultyColor(path.difficulty_level)} border text-xs`}>
                          {path.difficulty_level}
                        </Badge>
                        <div className={`text-xs font-medium ${getMatchColor(path.match_score)}`}>
                          {Math.round(path.match_score * 100)}% match
                        </div>
                      </div>
                    </div>

                    {/* Progress Bar */}
                    <div className="space-y-2 mb-3">
                      <div className="flex items-center justify-between text-sm">
                        <span className={progressStatus.color}>{progressStatus.status}</span>
                        <span className="font-medium">{Math.round(path.progress)}%</span>
                      </div>
                      <Progress value={path.progress} className="h-2" />
                    </div>

                    {/* Path Details */}
                    <div className="flex items-center justify-between text-sm">
                      <div className="flex items-center space-x-4">
                        <div className="flex items-center space-x-1 text-muted-foreground">
                          <Clock className="w-4 h-4" />
                          <span>{formatDuration(path.estimated_duration)}</span>
                        </div>
                        <div className="flex items-center space-x-1 text-muted-foreground">
                          <Users className="w-4 h-4" />
                          <span>{path.learning_modules?.length || 0} modules</span>
                        </div>
                      </div>
                      <ChevronRight className="w-4 h-4 text-muted-foreground" />
                    </div>

                    {/* Target Skills Preview */}
                    {path.target_skills?.length > 0 && (
                      <div className="mt-3 pt-3 border-t border-gray-100">
                        <div className="flex items-center space-x-2 mb-2">
                          <Target className="w-4 h-4 text-blue-600" />
                          <span className="text-sm font-medium">Target Skills</span>
                        </div>
                        <div className="flex flex-wrap gap-1">
                          {path.target_skills.slice(0, 3).map((skill, idx) => (
                            <Badge key={idx} variant="secondary" className="text-xs">
                              {skill}
                            </Badge>
                          ))}
                          {path.target_skills.length > 3 && (
                            <Badge variant="outline" className="text-xs">
                              +{path.target_skills.length - 3} more
                            </Badge>
                          )}
                        </div>
                      </div>
                    )}
                  </motion.div>
                );
              })}
            </AnimatePresence>
          </div>
        )}

        {/* Detailed Path View */}
        <AnimatePresence>
          {showDetails && selectedPath && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="border-t border-gray-200 pt-4 mt-4"
            >
              <div className="flex items-center justify-between mb-4">
                <h5 className="font-semibold">{selectedPath.title}</h5>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setShowDetails(false)}
                >
                  ×
                </Button>
              </div>

              <div className="space-y-4">
                {/* Career Impact */}
                {selectedPath.career_impact && (
                  <div className="bg-blue-50 rounded-lg p-3">
                    <div className="flex items-center space-x-2 mb-2">
                      <Star className="w-4 h-4 text-blue-600" />
                      <span className="font-medium text-blue-900">Career Impact</span>
                    </div>
                    <p className="text-sm text-blue-800">{selectedPath.career_impact}</p>
                  </div>
                )}

                {/* Learning Modules */}
                {selectedPath.learning_modules?.length > 0 && (
                  <div>
                    <h6 className="font-medium mb-3 flex items-center">
                      <BookOpen className="w-4 h-4 mr-2" />
                      Learning Modules ({selectedPath.learning_modules.length})
                    </h6>
                    <div className="space-y-2 max-h-40 overflow-y-auto">
                      {selectedPath.learning_modules.map((module, idx) => {
                        const ModuleIcon = getModuleIcon(module.type);
                        return (
                          <div key={module.id} className="flex items-center space-x-3 p-2 bg-gray-50 rounded">
                            <ModuleIcon className="w-4 h-4 text-gray-600" />
                            <div className="flex-1">
                              <div className="flex items-center justify-between">
                                <span className="text-sm font-medium">{module.title}</span>
                                {module.completed && (
                                  <CheckCircle className="w-4 h-4 text-green-600" />
                                )}
                              </div>
                              <div className="flex items-center space-x-2 text-xs text-muted-foreground">
                                <Clock className="w-3 h-3" />
                                <span>{module.duration}</span>
                                <span>•</span>
                                <span className="capitalize">{module.type}</span>
                              </div>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}

                {/* Completion Benefits */}
                {selectedPath.completion_benefits?.length > 0 && (
                  <div>
                    <h6 className="font-medium mb-2 flex items-center">
                      <Award className="w-4 h-4 mr-2" />
                      What You'll Gain
                    </h6>
                    <ul className="space-y-1">
                      {selectedPath.completion_benefits.map((benefit, idx) => (
                        <li key={idx} className="text-sm text-muted-foreground flex items-start">
                          <CheckCircle className="w-3 h-3 text-green-600 mr-2 mt-0.5 flex-shrink-0" />
                          {benefit}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Action Buttons */}
                <div className="flex items-center space-x-2 pt-3 border-t border-gray-100">
                  {selectedPath.progress === 0 ? (
                    <Button
                      onClick={() => onPathStart?.(selectedPath)}
                      className="flex items-center space-x-2"
                    >
                      <PlayCircle className="w-4 h-4" />
                      <span>Start Learning</span>
                    </Button>
                  ) : selectedPath.progress < 100 ? (
                    <Button
                      onClick={() => onPathContinue?.(selectedPath)}
                      className="flex items-center space-x-2"
                    >
                      <PlayCircle className="w-4 h-4" />
                      <span>Continue Learning</span>
                    </Button>
                  ) : (
                    <Button variant="secondary" className="flex items-center space-x-2">
                      <CheckCircle className="w-4 h-4" />
                      <span>Completed</span>
                    </Button>
                  )}
                  <Button variant="outline" size="sm">
                    View Details
                  </Button>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </CardContent>
    </Card>
  );
};

export default LearningPathsCard;