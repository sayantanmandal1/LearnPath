import React, { useState } from 'react';
import { motion } from 'motion/react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Textarea } from './ui/textarea';
import { Badge } from './ui/badge';
import { Progress } from './ui/progress';
import { 
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from './ui/select';
import { 
  Brain, 
  Target, 
  TrendingUp, 
  DollarSign, 
  Clock,
  Users,
  BookOpen,
  ArrowRight,
  Upload,
  FileText,
  Sparkles
} from 'lucide-react';

interface AnalysisData {
  jobTitle: string;
  experience: string;
  skills: string;
  location: string;
  interests: string;
  resume: File | null;
}

export function Analysis() {
  const [currentStep, setCurrentStep] = useState<number>(1);
  const [analysisData, setAnalysisData] = useState<AnalysisData>({
    jobTitle: '',
    experience: '',
    skills: '',
    location: '',
    interests: '',
    resume: null
  });
  const [showResults, setShowResults] = useState<boolean>(false);

  const experienceLevels = [
    'Entry Level (0-2 years)',
    'Mid Level (3-5 years)',
    'Senior Level (6-10 years)',
    'Expert Level (10+ years)'
  ];

  const locations = [
    'San Francisco, CA',
    'New York, NY',
    'Seattle, WA',
    'Austin, TX',
    'Boston, MA',
    'Remote',
    'Other'
  ];

  const handleAnalyze = () => {
    setShowResults(true);
    setCurrentStep(4);
  };

  const careerMatches = [
    {
      title: 'Frontend Developer',
      match: 92,
      salary: '$85,000 - $120,000',
      growth: '+15%',
      description: 'Perfect match for your React and JavaScript skills',
      skills: ['React', 'JavaScript', 'CSS', 'HTML'],
      inDemand: true
    },
    {
      title: 'Full Stack Developer',
      match: 78,
      salary: '$90,000 - $140,000',
      growth: '+12%',
      description: 'Great opportunity to expand backend skills',
      skills: ['Node.js', 'Python', 'Database', 'API'],
      inDemand: true
    },
    {
      title: 'UI/UX Designer',
      match: 65,
      salary: '$70,000 - $110,000',
      growth: '+8%',
      description: 'Leverage your frontend skills in design',
      skills: ['Figma', 'Design Systems', 'User Research'],
      inDemand: false
    }
  ];

  const skillGaps = [
    { skill: 'TypeScript', importance: 'High', currentLevel: 30, targetLevel: 80 },
    { skill: 'Node.js', importance: 'High', currentLevel: 20, targetLevel: 70 },
    { skill: 'Testing', importance: 'Medium', currentLevel: 40, targetLevel: 75 },
    { skill: 'DevOps', importance: 'Medium', currentLevel: 15, targetLevel: 60 },
  ];

  const learningPath = [
    {
      title: 'Complete TypeScript Fundamentals',
      description: 'Master TypeScript to improve code quality and developer experience',
      duration: '2 weeks',
      priority: 'High',
      type: 'Course'
    },
    {
      title: 'Build a Full Stack Project',
      description: 'Apply your skills in a real-world project with backend integration',
      duration: '4 weeks',
      priority: 'High',
      type: 'Project'
    },
    {
      title: 'Learn Testing Best Practices',
      description: 'Master unit and integration testing for React applications',
      duration: '1 week',
      priority: 'Medium',
      type: 'Course'
    }
  ];

  return (
    <div className="pt-16 min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto p-6">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="text-center mb-8"
        >
          <h1 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">
            Career Path Analysis
          </h1>
          <p className="text-xl text-gray-600 max-w-2xl mx-auto">
            Get personalized career recommendations based on your skills, experience, and goals
          </p>
        </motion.div>

        {!showResults ? (
          <div className="max-w-4xl mx-auto">
            {/* Progress Steps */}
            <div className="flex items-center justify-center mb-8">
              {[1, 2, 3].map((step) => (
                <div key={step} className="flex items-center">
                  <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
                    step <= currentStep ? 'bg-primary text-primary-foreground' : 'bg-gray-200 text-gray-600'
                  }`}>
                    {step}
                  </div>
                  {step < 3 && (
                    <div className={`w-16 h-1 mx-2 ${
                      step < currentStep ? 'bg-primary' : 'bg-gray-200'
                    }`} />
                  )}
                </div>
              ))}
            </div>

            <motion.div
              key={currentStep}
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5 }}
            >
              <Card>
                <CardHeader>
                  <CardTitle>
                    {currentStep === 1 && 'Basic Information'}
                    {currentStep === 2 && 'Skills & Experience'}
                    {currentStep === 3 && 'Goals & Preferences'}
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-6">
                  {currentStep === 1 && (
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div className="space-y-2">
                        <Label htmlFor="jobTitle">Current/Target Job Title</Label>
                        <Input
                          id="jobTitle"
                          placeholder="e.g., Frontend Developer"
                          value={analysisData.jobTitle}
                          onChange={(e) => setAnalysisData({ ...analysisData, jobTitle: e.target.value })}
                        />
                      </div>
                      <div className="space-y-2">
                        <Label htmlFor="experience">Experience Level</Label>
                        <Select value={analysisData.experience} onValueChange={(value: string) => setAnalysisData({ ...analysisData, experience: value })}>
                          <SelectTrigger>
                            <SelectValue placeholder="Select experience level" />
                          </SelectTrigger>
                          <SelectContent>
                            {experienceLevels.map((level) => (
                              <SelectItem key={level} value={level}>{level}</SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>
                      <div className="space-y-2">
                        <Label htmlFor="location">Preferred Location</Label>
                        <Select value={analysisData.location} onValueChange={(value: string) => setAnalysisData({ ...analysisData, location: value })}>
                          <SelectTrigger>
                            <SelectValue placeholder="Select location" />
                          </SelectTrigger>
                          <SelectContent>
                            {locations.map((location) => (
                              <SelectItem key={location} value={location}>{location}</SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>
                      <div className="space-y-2">
                        <Label htmlFor="resume">Upload Resume (Optional)</Label>
                        <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-gray-400 transition-colors cursor-pointer">
                          <Upload className="w-8 h-8 text-gray-400 mx-auto mb-2" />
                          <p className="text-sm text-gray-600">Click to upload or drag and drop</p>
                          <p className="text-xs text-gray-500">PDF, DOC up to 10MB</p>
                        </div>
                      </div>
                    </div>
                  )}

                  {currentStep === 2 && (
                    <div className="space-y-6">
                      <div className="space-y-2">
                        <Label htmlFor="skills">Technical Skills</Label>
                        <Textarea
                          id="skills"
                          placeholder="List your technical skills (e.g., JavaScript, React, Python, SQL...)"
                          rows={4}
                          value={analysisData.skills}
                          onChange={(e) => setAnalysisData({ ...analysisData, skills: e.target.value })}
                        />
                        <p className="text-sm text-gray-500">Separate skills with commas</p>
                      </div>
                      
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <Card className="p-4">
                          <div className="text-center">
                            <Brain className="w-8 h-8 text-blue-600 mx-auto mb-2" />
                            <h4 className="font-medium">AI Analysis</h4>
                            <p className="text-sm text-gray-600">We'll analyze your skills against market demand</p>
                          </div>
                        </Card>
                        <Card className="p-4">
                          <div className="text-center">
                            <Target className="w-8 h-8 text-green-600 mx-auto mb-2" />
                            <h4 className="font-medium">Skill Gaps</h4>
                            <p className="text-sm text-gray-600">Identify areas for improvement</p>
                          </div>
                        </Card>
                        <Card className="p-4">
                          <div className="text-center">
                            <TrendingUp className="w-8 h-8 text-purple-600 mx-auto mb-2" />
                            <h4 className="font-medium">Growth Path</h4>
                            <p className="text-sm text-gray-600">Personalized learning recommendations</p>
                          </div>
                        </Card>
                      </div>
                    </div>
                  )}

                  {currentStep === 3 && (
                    <div className="space-y-6">
                      <div className="space-y-2">
                        <Label htmlFor="interests">Career Interests & Goals</Label>
                        <Textarea
                          id="interests"
                          placeholder="Describe your career goals, interests, and what type of work environment you prefer..."
                          rows={4}
                          value={analysisData.interests}
                          onChange={(e) => setAnalysisData({ ...analysisData, interests: e.target.value })}
                        />
                      </div>

                      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div className="space-y-4">
                          <h4 className="font-medium">Work Environment Preferences</h4>
                          <div className="space-y-2">
                            {['Remote Work', 'Hybrid', 'On-site', 'Startup Environment', 'Large Corporation', 'Freelancing'].map((pref) => (
                              <div key={pref} className="flex items-center space-x-2">
                                <input type="checkbox" id={pref} className="rounded" />
                                <label htmlFor={pref} className="text-sm">{pref}</label>
                              </div>
                            ))}
                          </div>
                        </div>
                        
                        <div className="space-y-4">
                          <h4 className="font-medium">Industry Interests</h4>
                          <div className="space-y-2">
                            {['Technology', 'Healthcare', 'Finance', 'Education', 'E-commerce', 'Gaming'].map((industry) => (
                              <div key={industry} className="flex items-center space-x-2">
                                <input type="checkbox" id={industry} className="rounded" />
                                <label htmlFor={industry} className="text-sm">{industry}</label>
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>
                    </div>
                  )}

                  <div className="flex justify-between pt-6">
                    <Button 
                      variant="outline" 
                      onClick={() => setCurrentStep(Math.max(1, currentStep - 1))}
                      disabled={currentStep === 1}
                    >
                      Previous
                    </Button>
                    
                    {currentStep === 3 ? (
                      <Button onClick={handleAnalyze} className="bg-gradient-to-r from-blue-600 to-purple-600">
                        <Sparkles className="w-4 h-4 mr-2" />
                        Analyze My Career Path
                      </Button>
                    ) : (
                      <Button onClick={() => setCurrentStep(Math.min(3, currentStep + 1))}>
                        Next
                        <ArrowRight className="w-4 h-4 ml-2" />
                      </Button>
                    )}
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          </div>
        ) : (
          <div className="space-y-6">
            {/* Analysis Results */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
              className="text-center mb-8"
            >
              <Badge variant="secondary" className="mb-4 px-4 py-2">
                <Sparkles className="w-4 h-4 mr-2" />
                Analysis Complete
              </Badge>
              <h2 className="text-2xl font-bold text-gray-900 mb-2">
                Your Personalized Career Analysis
              </h2>
              <p className="text-gray-600">
                Based on your skills and goals, here are our recommendations
              </p>
            </motion.div>

            {/* Career Matches */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.1 }}
            >
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <Target className="w-5 h-5" />
                    <span>Top Career Matches</span>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-6">
                    {careerMatches.map((match, index) => (
                      <div key={index} className={`p-6 rounded-lg border-2 ${
                        index === 0 ? 'border-green-200 bg-green-50' : 'border-gray-200 bg-white'
                      }`}>
                        <div className="flex justify-between items-start mb-4">
                          <div>
                            <div className="flex items-center space-x-2 mb-2">
                              <h3 className="text-lg font-semibold">{match.title}</h3>
                              {match.inDemand && <Badge variant="secondary">High Demand</Badge>}
                              {index === 0 && <Badge className="bg-green-600">Best Match</Badge>}
                            </div>
                            <p className="text-gray-600 mb-2">{match.description}</p>
                          </div>
                          <div className="text-right">
                            <div className="text-2xl font-bold text-green-600">{match.match}%</div>
                            <div className="text-sm text-gray-500">Match</div>
                          </div>
                        </div>
                        
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                          <div className="flex items-center space-x-2">
                            <DollarSign className="w-4 h-4 text-gray-600" />
                            <span className="text-sm">{match.salary}</span>
                          </div>
                          <div className="flex items-center space-x-2">
                            <TrendingUp className="w-4 h-4 text-green-600" />
                            <span className="text-sm">Growth: {match.growth}</span>
                          </div>
                          <div className="flex items-center space-x-2">
                            <Users className="w-4 h-4 text-blue-600" />
                            <span className="text-sm">Market Demand</span>
                          </div>
                        </div>
                        
                        <div className="flex flex-wrap gap-2">
                          {match.skills.map((skill, skillIndex) => (
                            <Badge key={skillIndex} variant="outline">{skill}</Badge>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </motion.div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Skill Gaps */}
              <motion.div
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.6, delay: 0.2 }}
              >
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center space-x-2">
                      <Brain className="w-5 h-5" />
                      <span>Skill Gap Analysis</span>
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    {skillGaps.map((gap, index) => (
                      <div key={index} className="space-y-2">
                        <div className="flex justify-between items-center">
                          <span className="font-medium">{gap.skill}</span>
                          <Badge variant={gap.importance === 'High' ? 'destructive' : 'secondary'}>
                            {gap.importance}
                          </Badge>
                        </div>
                        <div className="space-y-1">
                          <div className="flex justify-between text-sm text-gray-600">
                            <span>Current: {gap.currentLevel}%</span>
                            <span>Target: {gap.targetLevel}%</span>
                          </div>
                          <div className="relative">
                            <Progress value={gap.currentLevel} className="h-2" />
                            <div 
                              className="absolute top-0 h-2 bg-green-200 rounded-full"
                              style={{ 
                                left: `${gap.currentLevel}%`, 
                                width: `${gap.targetLevel - gap.currentLevel}%` 
                              }}
                            />
                          </div>
                        </div>
                      </div>
                    ))}
                  </CardContent>
                </Card>
              </motion.div>

              {/* Learning Path */}
              <motion.div
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.6, delay: 0.3 }}
              >
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center space-x-2">
                      <BookOpen className="w-5 h-5" />
                      <span>Recommended Learning Path</span>
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    {learningPath.map((item, index) => (
                      <div key={index} className="p-4 border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors">
                        <div className="flex justify-between items-start mb-2">
                          <h4 className="font-medium">{item.title}</h4>
                          <Badge variant={item.priority === 'High' ? 'destructive' : 'secondary'}>
                            {item.priority}
                          </Badge>
                        </div>
                        <p className="text-sm text-gray-600 mb-3">{item.description}</p>
                        <div className="flex justify-between items-center">
                          <div className="flex items-center space-x-4 text-sm text-gray-500">
                            <div className="flex items-center space-x-1">
                              <Clock className="w-3 h-3" />
                              <span>{item.duration}</span>
                            </div>
                            <div className="flex items-center space-x-1">
                              <FileText className="w-3 h-3" />
                              <span>{item.type}</span>
                            </div>
                          </div>
                          <Button size="sm">Start Learning</Button>
                        </div>
                      </div>
                    ))}
                  </CardContent>
                </Card>
              </motion.div>
            </div>

            {/* Action Buttons */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.4 }}
              className="flex justify-center space-x-4"
            >
              <Button onClick={() => setShowResults(false)} variant="outline">
                New Analysis
              </Button>
              <Button>Save Results</Button>
              <Button variant="outline">Share Results</Button>
            </motion.div>
          </div>
        )}
      </div>
    </div>
  );
}