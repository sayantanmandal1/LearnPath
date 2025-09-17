import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Badge } from './ui/badge';
import { Avatar, AvatarImage, AvatarFallback } from './ui/avatar';
import { Separator } from './ui/separator';
import { Switch } from './ui/switch';
import { supabase } from '../utils/supabase/client';
import { toast } from 'sonner';
import {
  User,
  Mail,
  MapPin,
  Calendar,
  Edit,
  Link as LinkIcon,
  Settings,
  Shield,
  Bell,
  Trophy,
  Target,
  BookOpen,
  Award
} from 'lucide-react';

interface Achievement {
  title: string;
  description: string;
  date: string;
}

interface Activity {
  type: string;
  action: string;
  item: string;
  date: string;
}

export function Profile() {
  const [isEditing, setIsEditing] = useState(false);
  const [profile, setProfile] = useState({
    name: '',
    email: '',
    location: '',
    title: '',
    bio: '',
    website: '',
    career_goal: '',
    experience: '',
  });
  const [loading, setLoading] = useState(true);
  const [achievements, setAchievements] = useState<Achievement[]>([]);
  const [recentActivities, setRecentActivities] = useState<Activity[]>([]);
  const [learningStats, setLearningStats] = useState([
    { label: 'Courses Completed', value: 0, target: 0 },
    { label: 'Skills Mastered', value: 0, target: 0 },
    { label: 'Certificates Earned', value: 0, target: 0 },
    { label: 'Study Hours', value: 0, target: 0 },
  ]);

  // Delete account handler
  const handleDeleteAccount = async () => {
    if (!window.confirm('Are you sure you want to delete your account? This action cannot be undone.')) return;
    setLoading(true);
    try {
      const { data: { user }, error: userError } = await supabase.auth.getUser();
      if (userError || !user) {
        toast.error('Could not get user. Please log in again.');
        setLoading(false);
        return;
      }
      await supabase.from('profiles').delete().eq('id', user.id);
      // @ts-ignore
      await supabase.auth.admin.deleteUser(user.id);
      toast.success('Account deleted.');
      await supabase.auth.signOut();
      window.location.replace('/');
    } catch (err: any) {
      toast.error('Unexpected error: ' + (err.message || err));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    const fetchProfile = async () => {
      setLoading(true);
      const { data: { user }, error: userError } = await supabase.auth.getUser();
      if (userError || !user) {
        toast.error('Could not get user. Please log in again.');
        setLoading(false);
        return;
      }
      const { data, error } = await supabase.from('profiles').select('*').eq('id', user.id).single();
      if (!error && data) {
        setProfile({
          name: data.name || '',
          email: user.email || '',
          location: data.location || '',
          title: data.title || '',
          bio: data.bio || '',
          website: data.website || '',
          career_goal: data.career_goal || '',
          experience: data.experience || '',
        });
      } else {
        setProfile(p => ({ ...p, email: user.email || '' }));
      }
      setLoading(false);
    };
    fetchProfile();
  }, []);

  useEffect(() => {
    const fetchAchievements = async () => {
      const { data: { user }, error: userError } = await supabase.auth.getUser();
      if (userError || !user) return setAchievements([]);
      const { data, error } = await supabase
        .from('User Achievements')
        .select('*')
        .eq('user_id', user.id)
        .order('date', { ascending: false });
      setAchievements(!error && data ? data : []);
    };
    fetchAchievements();
  }, []);

  useEffect(() => {
    const fetchActivities = async () => {
      const { data: { user }, error: userError } = await supabase.auth.getUser();
      if (userError || !user) return setRecentActivities([]);
      const { data, error } = await supabase
        .from('User Activity Log')
        .select('*')
        .eq('user_id', user.id)
        .order('created_at', { ascending: false });
      setRecentActivities(!error && data ? data : []);
    };
    fetchActivities();
  }, []);

  useEffect(() => {
    const fetchStats = async () => {
      const { data: { user }, error: userError } = await supabase.auth.getUser();
      if (userError || !user) return;
      const { data, error } = await supabase
        .from('User Skills Table')
        .select('*')
        .eq('user_id', user.id)
        .single();
      if (!error && data) {
        setLearningStats([
          { label: 'Courses Completed', value: data.courses_completed || 0, target: data.courses_target || 0 },
          { label: 'Skills Mastered', value: data.skills_mastered || 0, target: data.skills_target || 0 },
          { label: 'Certificates Earned', value: data.certificates_earned || 0, target: data.certificates_target || 0 },
          { label: 'Study Hours', value: data.study_hours || 0, target: data.study_hours_target || 0 },
        ]);
      }
    };
    fetchStats();
  }, []);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) => {
    setProfile({ ...profile, [e.target.name]: e.target.value });
  };

  const handleSave = async () => {
    setLoading(true);
    try {
      const { data: { user }, error: userError } = await supabase.auth.getUser();
      if (userError || !user) {
        toast.error('Could not get user. Please log in again.');
        setLoading(false);
        return;
      }
      const { error } = await supabase.from('profiles').upsert({
        id: user.id,
        name: profile.name,
        location: profile.location,
        title: profile.title,
        bio: profile.bio,
        website: profile.website,
        career_goal: profile.career_goal,
        experience: profile.experience,
        updated_at: new Date().toISOString(),
      });
      if (error) {
        toast.error('Failed to save profile: ' + error.message);
      } else {
        toast.success('Profile updated!');
        setIsEditing(false);
      }
    } catch (err: any) {
      toast.error('Unexpected error: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return <div className="pt-16 min-h-screen flex items-center justify-center text-muted-foreground">Loading profile...</div>;
  }

  return (
    <div className="pt-16 min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto p-6 space-y-6">
        {/* Profile Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <Card>
            <CardContent className="p-6">
              <div className="flex flex-col md:flex-row items-start md:items-center space-y-4 md:space-y-0 md:space-x-6">
                <div className="relative">
                  <Avatar className="w-24 h-24">
                    <AvatarImage src="/avatars/user.png" alt="Profile" />
                    <AvatarFallback className="text-2xl">{profile.name ? profile.name[0] : 'U'}</AvatarFallback>
                  </Avatar>
                </div>
                <div className="flex-1">
                  {isEditing ? (
                    <div className="space-y-4">
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                          <Label htmlFor="name">Name</Label>
                          <Input id="name" name="name" value={profile.name} onChange={handleChange} />
                        </div>
                        <div>
                          <Label htmlFor="title">Title</Label>
                          <Input id="title" name="title" value={profile.title} onChange={handleChange} />
                        </div>
                        <div>
                          <Label htmlFor="location">Location</Label>
                          <Input id="location" name="location" value={profile.location} onChange={handleChange} />
                        </div>
                        <div>
                          <Label htmlFor="website">Website</Label>
                          <Input id="website" name="website" value={profile.website} onChange={handleChange} />
                        </div>
                        <div>
                          <Label htmlFor="career_goal">Career Goal</Label>
                          <Input id="career_goal" name="career_goal" value={profile.career_goal} onChange={handleChange} />
                        </div>
                        <div>
                          <Label htmlFor="experience">Experience</Label>
                          <select id="experience" name="experience" className="w-full border rounded px-3 py-2" value={profile.experience} onChange={handleChange}>
                            <option value="">Select experience</option>
                            <option value="student">Student</option>
                            <option value="entry">Entry Level</option>
                            <option value="mid">Mid Level</option>
                            <option value="senior">Senior</option>
                          </select>
                        </div>
                      </div>
                      <div>
                        <Label htmlFor="bio">Bio</Label>
                        <textarea id="bio" name="bio" className="w-full p-3 border border-border rounded-lg resize-none" rows={3} value={profile.bio} onChange={handleChange} />
                      </div>
                    </div>
                  ) : (
                    <div>
                      <h1 className="text-2xl font-bold text-gray-900">{profile.name}</h1>
                      <p className="text-lg text-gray-600">{profile.title}</p>
                      <p className="text-gray-500 mt-2">{profile.bio}</p>
                      <div className="flex flex-wrap items-center gap-4 mt-4 text-sm text-gray-500">
                        <div className="flex items-center space-x-1">
                          <Mail className="w-4 h-4" />
                          <span>{profile.email}</span>
                        </div>
                        <div className="flex items-center space-x-1">
                          <MapPin className="w-4 h-4" />
                          <span>{profile.location}</span>
                        </div>
                        <div className="flex items-center space-x-1">
                          <LinkIcon className="w-4 h-4" />
                          <span>{profile.website}</span>
                        </div>
                        <div className="flex items-center space-x-1">
                          <Calendar className="w-4 h-4" />
                          <span>Career Goal: {profile.career_goal}</span>
                        </div>
                        <div className="flex items-center space-x-1">
                          <Award className="w-4 h-4" />
                          <span>Experience: {profile.experience}</span>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
                <div className="flex space-x-2">
                  {isEditing ? (
                    <>
                      <Button onClick={handleSave} disabled={loading}>Save</Button>
                      <Button variant="outline" onClick={() => setIsEditing(false)} disabled={loading}>Cancel</Button>
                    </>
                  ) : (
                    <Button onClick={() => setIsEditing(true)}>
                      <Edit className="w-4 h-4 mr-2" />
                      Edit Profile
                    </Button>
                  )}
                </div>
              </div>
            </CardContent>
          </Card>
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Main Content */}
          <div className="lg:col-span-2 space-y-6">
            {/* Learning Stats */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6, delay: 0.2 }}
            >
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <BookOpen className="w-5 h-5" />
                    <span>Learning Progress</span>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 gap-4">
                    {learningStats.map((stat, index) => (
                      <div key={index} className="text-center p-4 bg-gray-50 rounded-lg">
                        <div className="text-2xl font-bold text-primary">{stat.value}</div>
                        <div className="text-xs text-muted-foreground">of {stat.target}</div>
                        <div className="text-sm font-medium mt-1">{stat.label}</div>
                        <div className="w-full bg-gray-200 rounded-full h-2 mt-2">
                          <div
                            className="bg-primary h-2 rounded-full transition-all duration-500"
                            style={{ width: `${stat.target ? (stat.value / stat.target) * 100 : 0}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </motion.div>

            {/* Achievements */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6, delay: 0.3 }}
            >
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <Trophy className="w-5 h-5" />
                    <span>Recent Achievements</span>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {achievements.map((achievement, index) => (
                      <div key={index} className="flex items-center space-x-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
                        <div className="p-2 bg-yellow-100 rounded-lg">
                          <Trophy className="w-5 h-5 text-yellow-600" />
                        </div>
                        <div className="flex-1">
                          <h4 className="font-medium text-gray-900">{achievement.title}</h4>
                          <p className="text-sm text-gray-600">{achievement.description}</p>
                          <p className="text-xs text-gray-500 mt-1">{achievement.date}</p>
                        </div>
                        <Badge variant="secondary">New</Badge>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </motion.div>

            {/* Recent Activities */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6, delay: 0.4 }}
            >
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <Calendar className="w-5 h-5" />
                    <span>Recent Activities</span>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {recentActivities.map((activity, index) => (
                      <div key={index} className="flex items-center space-x-4 p-3 hover:bg-gray-50 rounded-lg transition-colors">
                        <div className={`p-2 rounded-lg ${
                          activity.type === 'course' ? 'bg-blue-100' :
                          activity.type === 'achievement' ? 'bg-yellow-100' : 'bg-gray-100'
                        }`}>
                          {activity.type === 'course' ? (
                            <BookOpen className="w-4 h-4 text-blue-600" />
                          ) : activity.type === 'achievement' ? (
                            <Award className="w-4 h-4 text-yellow-600" />
                          ) : (
                            <User className="w-4 h-4 text-gray-600" />
                          )}
                        </div>
                        <div className="flex-1">
                          <p className="text-sm">
                            <span className="font-medium">{activity.action}</span> {activity.item}
                          </p>
                          <p className="text-xs text-muted-foreground">{activity.date}</p>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          </div>

          {/* Sidebar (add other widgets here if needed) */}
          <div className="space-y-6">
            {/* Settings */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6, delay: 0.3 }}
            >
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <Settings className="w-5 h-5" />
                    <span>Settings</span>
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <Bell className="w-4 h-4" />
                      <span className="text-sm">Email Notifications</span>
                    </div>
                    <Switch defaultChecked />
                  </div>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <Shield className="w-4 h-4" />
                      <span className="text-sm">Public Profile</span>
                    </div>
                    <Switch />
                  </div>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <Target className="w-4 h-4" />
                      <span className="text-sm">Weekly Goals</span>
                    </div>
                    <Switch defaultChecked />
                  </div>
                  <Separator />
                  <Button variant="outline" className="w-full">
                    Export Data
                  </Button>
                  <Button variant="destructive" className="w-full" onClick={handleDeleteAccount} disabled={loading}>
                    {loading ? 'Deleting...' : 'Delete Account'}
                  </Button>
                </CardContent>
              </Card>
            </motion.div>
          </div>
        </div>
      </div>
    </div>
  );
}