import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  Button,
  Badge,
  Progress,
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
  Alert,
  AlertDescription,
  Separator,
  ScrollArea
} from '@/components/ui';
import {
  Target,
  BookOpen,
  Calendar,
  Star,
  Clock,
  TrendingUp,
  CheckCircle,
  AlertCircle,
  ExternalLink,
  Download,
  Filter
} from 'lucide-react';

interface FocusArea {
  id: string;
  name: string;
  description: string;
  importance_score: number;
  current_level: string;
  target_level: string;
  skills_required: string[];
  estimated_time_weeks: number;
  priority_rank: number;
}

interface ProjectSpecification {
  id: string;
  title: string;
  description: string;
  difficulty_level: string;
  estimated_duration_weeks: number;
  technologies: string[];
  learning_outcomes: Array<{
    id: string;
    description: string;
    skills_gained: string[];
    competency_level: string;
    measurable_criteria: string[];
  }>;
  prerequisites: string[];
  deliverables: string[];
  success_metrics: string[];
  github_template_url?: string;
}

interface Milestone {
  id: string;
  title: string;
  description: string;
  target_date: string;
  completion_criteria: string[];
  dependencies: string[];
  estimated_effort_hours: number;
  resources_needed: string[];
}

interface PreparationRoadmap {
  id: string;
  target_role: string;
  total_duration_weeks: number;
  phases: Array<{
    id: string;
    name: string;
    description: string;
    start_week: number;
    duration_weeks: number;
    focus_area_id: string;
    objectives: string[];
  }>;
  milestones: Milestone[];
  critical_path: string[];
  buffer_time_weeks: number;
  success_probability: number;
}

interface CuratedResource {
  id: string;
  title: string;
  description: string;
  resource_type: string;
  url: string;
  provider: string;
  difficulty_level: string;
  estimated_time_hours: number;
  cost?: number;
  currency: string;
  rating: {
    overall_score: number;
    content_quality: number;
    difficulty_accuracy: number;
    practical_relevance: number;
    community_rating: number;
    last_updated: string;
  };
  tags: string[];
  prerequisites: string[];
  learning_outcomes: string[];
  is_free: boolean;
  certification_available: boolean;
}

interface CareerGuidanceData {
  user_id: string;
  target_role: string;
  generated_at: string;
  focus_areas: FocusArea[];
  project_specifications: ProjectSpecification[];
  preparation_roadmap: PreparationRoadmap;
  curated_resources: CuratedResource[];
  personalization_factors: Record<string, any>;
  confidence_score: number;
}

interface CareerGuidanceInterfaceProps {
  targetRole: string;
  currentExperience: number;
  timeCommitment: number;
  careerTimeline: number;
  onGuidanceGenerated?: (guidance: CareerGuidanceData) => void;
}

const CareerGuidanceInterface: React.FC<CareerGuidanceInterfaceProps> = ({
  targetRole,
  currentExperience,
  timeCommitment,
  careerTimeline,
  onGuidanceGenerated
}) => {
  const [guidanceData, setGuidanceData] = useState<CareerGuidanceData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState('focus-areas');
  const [selectedFocusArea, setSelectedFocusArea] = useState<string | null>(null);
  const [resourceFilter, setResourceFilter] = useState<string>('all');

  useEffect(() => {
    generateCareerGuidance();
  }, [targetRole, currentExperience, timeCommitment, careerTimeline]);

  const generateCareerGuidance = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/v1/career-guidance/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify({
          target_role: targetRole,
          current_experience_years: currentExperience,
          time_commitment_hours_per_week: timeCommitment,
          career_timeline_months: careerTimeline,
          preferred_learning_style: 'hands-on',
          specific_interests: []
        })
      });

      if (!response.ok) {
        throw new Error('Failed to generate career guidance');
      }

      const data = await response.json();
      setGuidanceData(data);
      onGuidanceGenerated?.(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const getDifficultyColor = (level: string) => {
    switch (level.toLowerCase()) {
      case 'beginner': return 'bg-green-100 text-green-800';
      case 'intermediate': return 'bg-yellow-100 text-yellow-800';
      case 'advanced': return 'bg-orange-100 text-orange-800';
      case 'expert': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getResourceTypeIcon = (type: string) => {
    switch (type.toLowerCase()) {
      case 'course': return <BookOpen className="w-4 h-4" />;
      case 'book': return <BookOpen className="w-4 h-4" />;
      case 'tutorial': return <Target className="w-4 h-4" />;
      case 'documentation': return <BookOpen className="w-4 h-4" />;
      case 'project': return <Target className="w-4 h-4" />;
      case 'practice': return <TrendingUp className="w-4 h-4" />;
      case 'certification': return <Star className="w-4 h-4" />;
      default: return <BookOpen className="w-4 h-4" />;
    }
  };

  const renderStarRating = (rating: number) => {
    return (
      <div className="flex items-center space-x-1">
        {[1, 2, 3, 4, 5].map((star) => (
          <Star
            key={star}
            className={`w-4 h-4 ${
              star <= rating ? 'text-yellow-400 fill-current' : 'text-gray-300'
            }`}
          />
        ))}
        <span className="text-sm text-gray-600 ml-1">{rating.toFixed(1)}</span>
      </div>
    );
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center p-8">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Generating your personalized career guidance...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <Alert className="m-4">
        <AlertCircle className="h-4 w-4" />
        <AlertDescription>
          {error}
          <Button onClick={generateCareerGuidance} className="ml-4" size="sm">
            Retry
          </Button>
        </AlertDescription>
      </Alert>
    );
  }

  if (!guidanceData) {
    return null;
  }

  return (
    <div className="max-w-7xl mx-auto p-6 space-y-6">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center mb-8"
      >
        <h1 className="text-3xl font-bold text-gray-900 mb-2">
          Career Guidance for {guidanceData.target_role}
        </h1>
        <p className="text-gray-600 mb-4">
          Personalized roadmap generated on {new Date(guidanceData.generated_at).toLocaleDateString()}
        </p>
        <div className="flex items-center justify-center space-x-4">
          <Badge variant="outline" className="px-3 py-1">
            <TrendingUp className="w-4 h-4 mr-1" />
            {Math.round(guidanceData.confidence_score * 100)}% Confidence
          </Badge>
          <Badge variant="outline" className="px-3 py-1">
            <Calendar className="w-4 h-4 mr-1" />
            {guidanceData.preparation_roadmap.total_duration_weeks} weeks
          </Badge>
          <Badge variant="outline" className="px-3 py-1">
            <Clock className="w-4 h-4 mr-1" />
            {timeCommitment}h/week
          </Badge>
        </div>
      </motion.div>

      {/* Main Content Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="focus-areas">Focus Areas</TabsTrigger>
          <TabsTrigger value="projects">Projects</TabsTrigger>
          <TabsTrigger value="roadmap">Roadmap</TabsTrigger>
          <TabsTrigger value="resources">Resources</TabsTrigger>
        </TabsList>

        {/* Focus Areas Tab */}
        <TabsContent value="focus-areas" className="space-y-6">
          <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
            {guidanceData.focus_areas.map((area, index) => (
              <motion.div
                key={area.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
              >
                <Card className="h-full hover:shadow-lg transition-shadow cursor-pointer"
                      onClick={() => setSelectedFocusArea(area.id)}>
                  <CardHeader>
                    <div className="flex items-start justify-between">
                      <CardTitle className="text-lg">{area.name}</CardTitle>
                      <Badge className="ml-2">#{area.priority_rank}</Badge>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Badge className={getDifficultyColor(area.current_level)}>
                        Current: {area.current_level}
                      </Badge>
                      <Badge className={getDifficultyColor(area.target_level)}>
                        Target: {area.target_level}
                      </Badge>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <p className="text-gray-600 mb-4">{area.description}</p>
                    
                    <div className="space-y-3">
                      <div>
                        <div className="flex justify-between text-sm mb-1">
                          <span>Importance Score</span>
                          <span>{area.importance_score}/10</span>
                        </div>
                        <Progress value={area.importance_score * 10} className="h-2" />
                      </div>
                      
                      <div>
                        <p className="text-sm font-medium mb-2">Skills Required:</p>
                        <div className="flex flex-wrap gap-1">
                          {area.skills_required.map((skill) => (
                            <Badge key={skill} variant="secondary" className="text-xs">
                              {skill}
                            </Badge>
                          ))}
                        </div>
                      </div>
                      
                      <div className="flex items-center text-sm text-gray-600">
                        <Clock className="w-4 h-4 mr-1" />
                        {area.estimated_time_weeks} weeks estimated
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            ))}
          </div>
        </TabsContent>

        {/* Projects Tab */}
        <TabsContent value="projects" className="space-y-6">
          <div className="grid gap-6 lg:grid-cols-2">
            {guidanceData.project_specifications.map((project, index) => (
              <motion.div
                key={project.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
              >
                <Card className="h-full">
                  <CardHeader>
                    <div className="flex items-start justify-between">
                      <CardTitle className="text-xl">{project.title}</CardTitle>
                      <Badge className={getDifficultyColor(project.difficulty_level)}>
                        {project.difficulty_level}
                      </Badge>
                    </div>
                    <p className="text-gray-600">{project.description}</p>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div className="flex items-center">
                        <Calendar className="w-4 h-4 mr-2 text-gray-500" />
                        {project.estimated_duration_weeks} weeks
                      </div>
                      <div className="flex items-center">
                        <Target className="w-4 h-4 mr-2 text-gray-500" />
                        {project.learning_outcomes.length} outcomes
                      </div>
                    </div>

                    <div>
                      <p className="font-medium mb-2">Technologies:</p>
                      <div className="flex flex-wrap gap-1">
                        {project.technologies.map((tech) => (
                          <Badge key={tech} variant="outline" className="text-xs">
                            {tech}
                          </Badge>
                        ))}
                      </div>
                    </div>

                    <div>
                      <p className="font-medium mb-2">Learning Outcomes:</p>
                      <ul className="space-y-1">
                        {project.learning_outcomes.slice(0, 3).map((outcome) => (
                          <li key={outcome.id} className="text-sm text-gray-600 flex items-start">
                            <CheckCircle className="w-4 h-4 mr-2 text-green-500 mt-0.5 flex-shrink-0" />
                            {outcome.description}
                          </li>
                        ))}
                      </ul>
                    </div>

                    <div>
                      <p className="font-medium mb-2">Key Deliverables:</p>
                      <ul className="space-y-1">
                        {project.deliverables.slice(0, 3).map((deliverable, idx) => (
                          <li key={idx} className="text-sm text-gray-600 flex items-start">
                            <Target className="w-4 h-4 mr-2 text-blue-500 mt-0.5 flex-shrink-0" />
                            {deliverable}
                          </li>
                        ))}
                      </ul>
                    </div>

                    {project.github_template_url && (
                      <Button variant="outline" className="w-full" asChild>
                        <a href={project.github_template_url} target="_blank" rel="noopener noreferrer">
                          <ExternalLink className="w-4 h-4 mr-2" />
                          View Template
                        </a>
                      </Button>
                    )}
                  </CardContent>
                </Card>
              </motion.div>
            ))}
          </div>
        </TabsContent>

        {/* Roadmap Tab */}
        <TabsContent value="roadmap" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center">
                <Calendar className="w-5 h-5 mr-2" />
                Preparation Roadmap
              </CardTitle>
              <div className="flex items-center space-x-4 text-sm text-gray-600">
                <span>Total Duration: {guidanceData.preparation_roadmap.total_duration_weeks} weeks</span>
                <span>Buffer Time: {guidanceData.preparation_roadmap.buffer_time_weeks} weeks</span>
                <span>Success Probability: {Math.round(guidanceData.preparation_roadmap.success_probability * 100)}%</span>
              </div>
            </CardHeader>
            <CardContent>
              {/* Phases */}
              <div className="space-y-6">
                <h3 className="text-lg font-semibold">Learning Phases</h3>
                <div className="space-y-4">
                  {guidanceData.preparation_roadmap.phases.map((phase, index) => (
                    <motion.div
                      key={phase.id}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.1 }}
                      className="border rounded-lg p-4"
                    >
                      <div className="flex items-start justify-between mb-2">
                        <h4 className="font-medium">{phase.name}</h4>
                        <Badge variant="outline">
                          Week {phase.start_week} - {phase.start_week + phase.duration_weeks - 1}
                        </Badge>
                      </div>
                      <p className="text-gray-600 mb-3">{phase.description}</p>
                      <div>
                        <p className="text-sm font-medium mb-1">Objectives:</p>
                        <div className="flex flex-wrap gap-1">
                          {phase.objectives.map((objective) => (
                            <Badge key={objective} variant="secondary" className="text-xs">
                              {objective}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    </motion.div>
                  ))}
                </div>
              </div>

              <Separator className="my-6" />

              {/* Milestones */}
              <div className="space-y-6">
                <h3 className="text-lg font-semibold">Key Milestones</h3>
                <div className="space-y-4">
                  {guidanceData.preparation_roadmap.milestones.map((milestone, index) => (
                    <motion.div
                      key={milestone.id}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: index * 0.1 }}
                      className="border rounded-lg p-4"
                    >
                      <div className="flex items-start justify-between mb-2">
                        <h4 className="font-medium">{milestone.title}</h4>
                        <div className="text-right">
                          <Badge variant="outline" className="mb-1">
                            {new Date(milestone.target_date).toLocaleDateString()}
                          </Badge>
                          <div className="text-xs text-gray-500">
                            {milestone.estimated_effort_hours}h effort
                          </div>
                        </div>
                      </div>
                      <p className="text-gray-600 mb-3">{milestone.description}</p>
                      
                      <div className="grid md:grid-cols-2 gap-4">
                        <div>
                          <p className="text-sm font-medium mb-1">Completion Criteria:</p>
                          <ul className="space-y-1">
                            {milestone.completion_criteria.map((criteria, idx) => (
                              <li key={idx} className="text-sm text-gray-600 flex items-start">
                                <CheckCircle className="w-3 h-3 mr-2 text-green-500 mt-1 flex-shrink-0" />
                                {criteria}
                              </li>
                            ))}
                          </ul>
                        </div>
                        
                        <div>
                          <p className="text-sm font-medium mb-1">Resources Needed:</p>
                          <div className="flex flex-wrap gap-1">
                            {milestone.resources_needed.map((resource) => (
                              <Badge key={resource} variant="secondary" className="text-xs">
                                {resource}
                              </Badge>
                            ))}
                          </div>
                        </div>
                      </div>
                    </motion.div>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Resources Tab */}
        <TabsContent value="resources" className="space-y-6">
          <div className="flex items-center justify-between">
            <h2 className="text-2xl font-bold">Curated Learning Resources</h2>
            <div className="flex items-center space-x-2">
              <Filter className="w-4 h-4" />
              <select
                value={resourceFilter}
                onChange={(e) => setResourceFilter(e.target.value)}
                className="border rounded px-3 py-1 text-sm"
              >
                <option value="all">All Resources</option>
                <option value="free">Free Only</option>
                <option value="course">Courses</option>
                <option value="book">Books</option>
                <option value="certification">Certifications</option>
              </select>
            </div>
          </div>

          <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
            {guidanceData.curated_resources
              .filter(resource => {
                if (resourceFilter === 'all') return true;
                if (resourceFilter === 'free') return resource.is_free;
                return resource.resource_type.toLowerCase() === resourceFilter;
              })
              .map((resource, index) => (
                <motion.div
                  key={resource.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.1 }}
                >
                  <Card className="h-full hover:shadow-lg transition-shadow">
                    <CardHeader>
                      <div className="flex items-start justify-between">
                        <div className="flex items-center space-x-2">
                          {getResourceTypeIcon(resource.resource_type)}
                          <CardTitle className="text-lg">{resource.title}</CardTitle>
                        </div>
                        {resource.is_free && (
                          <Badge className="bg-green-100 text-green-800">Free</Badge>
                        )}
                      </div>
                      <div className="flex items-center justify-between">
                        <Badge className={getDifficultyColor(resource.difficulty_level)}>
                          {resource.difficulty_level}
                        </Badge>
                        {renderStarRating(resource.rating.overall_score)}
                      </div>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <p className="text-gray-600 text-sm">{resource.description}</p>
                      
                      <div className="grid grid-cols-2 gap-2 text-sm">
                        <div className="flex items-center">
                          <Clock className="w-4 h-4 mr-1 text-gray-500" />
                          {resource.estimated_time_hours}h
                        </div>
                        <div className="flex items-center">
                          <span className="text-gray-500">By {resource.provider}</span>
                        </div>
                      </div>

                      {!resource.is_free && resource.cost && (
                        <div className="text-sm font-medium">
                          ${resource.cost} {resource.currency}
                        </div>
                      )}

                      <div>
                        <p className="text-sm font-medium mb-1">Tags:</p>
                        <div className="flex flex-wrap gap-1">
                          {resource.tags.slice(0, 4).map((tag) => (
                            <Badge key={tag} variant="secondary" className="text-xs">
                              {tag}
                            </Badge>
                          ))}
                        </div>
                      </div>

                      <div>
                        <p className="text-sm font-medium mb-1">You'll Learn:</p>
                        <ul className="space-y-1">
                          {resource.learning_outcomes.slice(0, 2).map((outcome, idx) => (
                            <li key={idx} className="text-xs text-gray-600 flex items-start">
                              <CheckCircle className="w-3 h-3 mr-1 text-green-500 mt-0.5 flex-shrink-0" />
                              {outcome}
                            </li>
                          ))}
                        </ul>
                      </div>

                      <div className="flex space-x-2">
                        <Button variant="outline" size="sm" className="flex-1" asChild>
                          <a href={resource.url} target="_blank" rel="noopener noreferrer">
                            <ExternalLink className="w-4 h-4 mr-1" />
                            View Resource
                          </a>
                        </Button>
                        {resource.certification_available && (
                          <Button variant="outline" size="sm">
                            <Star className="w-4 h-4" />
                          </Button>
                        )}
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>
              ))}
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default CareerGuidanceInterface;