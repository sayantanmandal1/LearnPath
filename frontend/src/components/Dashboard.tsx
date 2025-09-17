// TypeScript interfaces for dashboard data
interface Activity {
  action: string;
  item: string;
  time: string;
  score?: number | null;
}
interface Skill {
  name: string;
  progress: number;
  level: 'Beginner' | 'Intermediate' | 'Advanced';
}
interface CareerPath {
  name: string;
  value: number;
  color: string;
}
interface Recommendation {
  title: string;
  description: string;
  priority: 'High' | 'Medium' | 'Low';
  timeEstimate: string;
  icon: React.ElementType;
}
import React, { useEffect, useState } from 'react';
import { motion } from 'motion/react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import { Progress } from './ui/progress';
import { Avatar, AvatarImage, AvatarFallback } from './ui/avatar';
import { supabase } from '../utils/supabase/client';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell
} from 'recharts';
import { 
  TrendingUp, 
  Target, 
  BookOpen, 
  Award, 
  Clock, 
  Users,
  ArrowUpRight,
  Brain,
  Star,
  Calendar,
  Trophy
} from 'lucide-react';

export function Dashboard() {
  // Achievements state
  const [user, setUser] = useState<any>(null);
  const [achievements, setAchievements] = useState<any[]>([]);
  const [loadingAchievements, setLoadingAchievements] = useState(true);

  useEffect(() => {
    const getUser = async () => {
      const { data: { user } } = await supabase.auth.getUser();
      setUser(user);
    };
    getUser();
  }, []);

  useEffect(() => {
    const fetchAchievements = async () => {
      if (!user) return;
      setLoadingAchievements(true);
      const { data, error } = await supabase
        .from('achievements')
        .select('*')
        .eq('user_id', user.id)
        .order('date', { ascending: false });
      if (!error && data) {
        setAchievements(data);
      } else {
        setAchievements([]);
      }
      setLoadingAchievements(false);
    };
    fetchAchievements();
  }, [user]);

  // Activities and skills state
  const [activityData, setActivityData] = useState<any[]>([]); // keep as any[] unless you have a schema
  const [recentActivities, setRecentActivities] = useState<Activity[]>([]);
  const [skillData, setSkillData] = useState<Skill[]>([]);
  const [loadingSkills, setLoadingSkills] = useState(true);

  useEffect(() => {
    const fetchSkills = async () => {
      if (!user) return;
      setLoadingSkills(true);
      const { data, error } = await supabase
        .from('skills')
        .select('*')
        .eq('user_id', user.id);
      if (!error && data) {
        setSkillData(data);
      } else {
        setSkillData([]);
      }
      setLoadingSkills(false);
    };
    fetchSkills();
  }, [user]);
  const [careerPathData, setCareerPathData] = useState<CareerPath[]>([
    { name: 'Frontend Developer', value: 40, color: '#3b82f6' },
    { name: 'Full Stack Developer', value: 30, color: '#10b981' },
    { name: 'Data Scientist', value: 20, color: '#f59e0b' },
    { name: 'DevOps Engineer', value: 10, color: '#ef4444' },
  ]);
  const [recommendations, setRecommendations] = useState<Recommendation[]>([
    {
      title: 'Complete React Advanced Course',
      description: 'Master advanced React patterns and state management',
      priority: 'High',
      timeEstimate: '2 weeks',
      icon: BookOpen
    },
    {
      title: 'Learn TypeScript Fundamentals',
      description: 'Add type safety to your JavaScript development',
      priority: 'Medium',
      timeEstimate: '1 week',
      icon: Target
    },
    {
      title: 'Build a Full Stack Project',
      description: 'Apply your skills in a real-world project',
      priority: 'High',
      timeEstimate: '4 weeks',
      icon: Award
    }
  ]);
  const [loadingActivities, setLoadingActivities] = useState(true);

  useEffect(() => {
    const fetchUserData = async () => {
      if (!user) return;
      setLoadingActivities(true);
      // Fetch activities for this user from Supabase
      const { data: activities, error } = await supabase
        .from('activities')
        .select('*')
        .eq('user_id', user.id)
        .order('created_at', { ascending: false });
      if (!error && activities) {
        setRecentActivities(activities.map((a: any) => ({
          action: a.action,
          item: a.item,
          time: a.time || a.created_at,
          score: a.score || null
        })));
        // Optionally, set activityData, skillData, etc. from activities if your schema supports it
      } else {
        setRecentActivities([]);
      }
      setLoadingActivities(false);
    };
    fetchUserData();
  }, [user]);



  useEffect(() => {
    const fetchUserData = async () => {
      if (!user) return;
      setLoadingActivities(true);
      // Fetch activities for this user from Supabase
      const { data: activities, error } = await supabase
        .from('activities')
        .select('*')
        .eq('user_id', user.id)
        .order('created_at', { ascending: false });
      if (!error && activities) {
        setRecentActivities(activities.map((a: any) => ({
          action: a.action,
          item: a.item,
          time: a.time || a.created_at,
          score: a.score || null
        })));
        // Optionally, set activityData, skillData, etc. from activities if your schema supports it
      } else {
        setRecentActivities([]);
      }
      setLoadingActivities(false);
    };
    fetchUserData();
  }, [user]);

  return (
    <div className="pt-16 min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto p-6 space-y-6">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="flex flex-col md:flex-row justify-between items-start md:items-center space-y-4 md:space-y-0"
        >
          <div>
            <h1 className="text-3xl font-bold text-gray-900">
              Welcome back, {user?.user_metadata?.name || user?.email?.split('@')[0] || 'User'}!
            </h1>
            <p className="text-gray-600 mt-1">Here's your career progress overview</p>
          </div>
          <Button className="flex items-center space-x-2">
            <Brain className="w-4 h-4" />
            <span>Get AI Insights</span>
          </Button>
        </motion.div>

        {/* Key Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {[
            { title: 'Skill Level', value: '75%', change: '+5%', icon: TrendingUp, color: 'text-green-600' },
            { title: 'Courses Completed', value: '12', change: '+3', icon: BookOpen, color: 'text-blue-600' },
            { title: 'Career Match', value: '87%', change: '+12%', icon: Target, color: 'text-purple-600' },
            { title: 'Learning Streak', value: '15 days', change: '+2', icon: Award, color: 'text-orange-600' },
          ].map((metric, index) => {
            const Icon = metric.icon;
            return (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
              >
                <Card>
                  <CardContent className="p-6">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm text-muted-foreground">{metric.title}</p>
                        <p className="text-2xl font-bold">{metric.value}</p>
                        <p className={`text-sm ${metric.color} flex items-center mt-1`}>
                          <ArrowUpRight className="w-3 h-3 mr-1" />
                          {metric.change}
                        </p>
                      </div>
                      <div className={`p-3 rounded-lg bg-gray-100`}>
                        <Icon className="w-6 h-6 text-gray-600" />
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            );
          })}
        </div>


        {/* Charts Section */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Achievements */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.3 }}
          >
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Award className="w-5 h-5" />
                  <span>Recent Achievements</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {loadingAchievements ? (
                    <div className="text-center text-muted-foreground py-8">Loading achievements...</div>
                  ) : achievements.length === 0 ? (
                    <div className="text-center text-muted-foreground py-8">No achievements yet.</div>
                  ) : achievements.map((achievement, index) => {
                    let Icon = Award;
                    if (achievement.icon === 'Target') Icon = Target;
                    if (achievement.icon === 'Trophy') Icon = Trophy;
                    // Add more icon mappings as needed
                    return (
                      <div key={achievement.id || index} className="flex items-center space-x-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
                        <div className="p-2 bg-yellow-100 rounded-lg">
                          <Icon className="w-5 h-5 text-yellow-600" />
                        </div>
                        <div className="flex-1">
                          <h4 className="font-medium text-gray-900">{achievement.title}</h4>
                          <p className="text-sm text-gray-600">{achievement.description}</p>
                          <p className="text-xs text-gray-500 mt-1">{achievement.date}</p>
                        </div>
                        <Badge variant="secondary">New</Badge>
                      </div>
                    );
                  })}
                </div>
              </CardContent>
            </Card>
          </motion.div>

          {/* Career Path Distribution */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.4 }}
          >
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Target className="w-5 h-5" />
                  <span>Career Path Match</span>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={careerPathData}
                      cx="50%"
                      cy="50%"
                      outerRadius={80}
                      dataKey="value"
                      label={({name, value}) => `${name}: ${value}%`}
                    >
                      {careerPathData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </motion.div>
        </div>

        {/* Skills and Recommendations */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Skill Progress */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.5 }}
          >
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Star className="w-5 h-5" />
                  <span>Skill Progress</span>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                {loadingSkills ? (
                  <div className="text-center text-muted-foreground py-8">Loading skills...</div>
                ) : skillData.length === 0 ? (
                  <div className="text-center text-muted-foreground py-8">No skills yet. Add your first skill!</div>
                ) : skillData.map((skill, index) => (
                  <div key={index} className="space-y-2">
                    <div className="flex justify-between items-center">
                      <span className="font-medium">{skill.name}</span>
                      <Badge variant={skill.level === 'Advanced' ? 'default' : skill.level === 'Intermediate' ? 'secondary' : 'outline'}>
                        {skill.level}
                      </Badge>
                    </div>
                    <Progress value={skill.progress} className="h-2" />
                    <div className="text-sm text-muted-foreground">{skill.progress}% Complete</div>
                  </div>
                ))}
              </CardContent>
            </Card>
          </motion.div>

          {/* Recommendations */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.6 }}
          >
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Brain className="w-5 h-5" />
                  <span>AI Recommendations</span>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                {recommendations.map((rec, index) => {
                  const Icon = rec.icon;
                  return (
                    <div key={index} className="p-4 border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors">
                      <div className="flex items-start space-x-3">
                        <div className="p-2 bg-blue-100 rounded-lg">
                          <Icon className="w-4 h-4 text-blue-600" />
                        </div>
                        <div className="flex-1">
                          <div className="flex items-center justify-between mb-1">
                            <h4 className="font-medium">{rec.title}</h4>
                            <Badge variant={rec.priority === 'High' ? 'destructive' : 'secondary'}>
                              {rec.priority}
                            </Badge>
                          </div>
                          <p className="text-sm text-muted-foreground mb-2">{rec.description}</p>
                          <div className="flex items-center text-xs text-muted-foreground">
                            <Clock className="w-3 h-3 mr-1" />
                            {rec.timeEstimate}
                          </div>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </CardContent>
            </Card>
          </motion.div>
        </div>

        {/* Recent Activity */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.7 }}
        >
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Calendar className="w-5 h-5" />
                <span>Recent Activity</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {loadingActivities ? (
                  <div className="text-center text-muted-foreground py-8">Loading activities...</div>
                ) : recentActivities.length === 0 ? (
                  <div className="text-center text-muted-foreground py-8">No activities yet. Start learning to see your progress here!</div>
                ) : recentActivities.map((activity, index) => (
                  <div key={index} className="flex items-center space-x-4 p-3 hover:bg-gray-50 rounded-lg transition-colors">
                    <Avatar className="w-8 h-8">
                      <AvatarImage src={user?.user_metadata?.avatar_url} />
                      <AvatarFallback className="text-xs">
                        {user?.email?.charAt(0).toUpperCase() || 'U'}
                      </AvatarFallback>
                    </Avatar>
                    <div className="flex-1">
                      <p className="text-sm">
                        <span className="font-medium">{activity.action}</span> {activity.item}
                        {activity.score && (
                          <Badge variant="secondary" className="ml-2">
                            {activity.score}%
                          </Badge>
                        )}
                      </p>
                      <p className="text-xs text-muted-foreground">{activity.time}</p>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </motion.div>
      </div>
    </div>
  );
}