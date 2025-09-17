-- Supabase Database Setup for Career AI Platform
-- Run this in your Supabase SQL Editor

-- Enable Row Level Security
ALTER TABLE auth.users ENABLE ROW LEVEL SECURITY;

-- Create user profiles table
CREATE TABLE IF NOT EXISTS public.user_profiles (
    id UUID REFERENCES auth.users(id) ON DELETE CASCADE PRIMARY KEY,
    email TEXT,
    full_name TEXT,
    first_name TEXT,
    last_name TEXT,
    career_goal TEXT,
    experience_level TEXT,
    current_role TEXT,
    location TEXT,
    github_username TEXT,
    leetcode_id TEXT,
    linkedin_url TEXT,
    skills TEXT[],
    bio TEXT,
    avatar_url TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create achievements table
CREATE TABLE IF NOT EXISTS public.achievements (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    description TEXT,
    category TEXT,
    points INTEGER DEFAULT 0,
    icon TEXT,
    date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create skills table
CREATE TABLE IF NOT EXISTS public.skills (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    level TEXT CHECK (level IN ('Beginner', 'Intermediate', 'Advanced')),
    progress INTEGER DEFAULT 0 CHECK (progress >= 0 AND progress <= 100),
    category TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create activities table
CREATE TABLE IF NOT EXISTS public.activities (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    action TEXT NOT NULL,
    item TEXT NOT NULL,
    score INTEGER,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create career_paths table
CREATE TABLE IF NOT EXISTS public.career_paths (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    value INTEGER DEFAULT 0,
    color TEXT,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create learning_recommendations table
CREATE TABLE IF NOT EXISTS public.learning_recommendations (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    description TEXT,
    priority TEXT CHECK (priority IN ('High', 'Medium', 'Low')),
    time_estimate TEXT,
    category TEXT,
    url TEXT,
    completed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create job_recommendations table
CREATE TABLE IF NOT EXISTS public.job_recommendations (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    company TEXT,
    location TEXT,
    salary_range TEXT,
    match_percentage INTEGER,
    description TEXT,
    requirements TEXT[],
    url TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Row Level Security Policies
-- User profiles
CREATE POLICY "Users can view own profile" ON public.user_profiles
    FOR SELECT USING (auth.uid() = id);

CREATE POLICY "Users can update own profile" ON public.user_profiles
    FOR UPDATE USING (auth.uid() = id);

CREATE POLICY "Users can insert own profile" ON public.user_profiles
    FOR INSERT WITH CHECK (auth.uid() = id);

-- Achievements
CREATE POLICY "Users can view own achievements" ON public.achievements
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own achievements" ON public.achievements
    FOR INSERT WITH CHECK (auth.uid() = user_id);

-- Skills
CREATE POLICY "Users can view own skills" ON public.skills
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own skills" ON public.skills
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own skills" ON public.skills
    FOR UPDATE USING (auth.uid() = user_id);

-- Activities
CREATE POLICY "Users can view own activities" ON public.activities
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own activities" ON public.activities
    FOR INSERT WITH CHECK (auth.uid() = user_id);

-- Career paths
CREATE POLICY "Users can view own career paths" ON public.career_paths
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own career paths" ON public.career_paths
    FOR INSERT WITH CHECK (auth.uid() = user_id);

-- Learning recommendations
CREATE POLICY "Users can view own learning recommendations" ON public.learning_recommendations
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own learning recommendations" ON public.learning_recommendations
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own learning recommendations" ON public.learning_recommendations
    FOR UPDATE USING (auth.uid() = user_id);

-- Job recommendations
CREATE POLICY "Users can view own job recommendations" ON public.job_recommendations
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own job recommendations" ON public.job_recommendations
    FOR INSERT WITH CHECK (auth.uid() = user_id);

-- Enable RLS on all tables
ALTER TABLE public.user_profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.achievements ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.skills ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.activities ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.career_paths ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.learning_recommendations ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.job_recommendations ENABLE ROW LEVEL SECURITY;

-- Function to handle user profile creation
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO public.user_profiles (id, email, full_name, first_name, last_name, career_goal, experience_level)
    VALUES (
        NEW.id,
        NEW.email,
        NEW.raw_user_meta_data->>'full_name',
        NEW.raw_user_meta_data->>'first_name',
        NEW.raw_user_meta_data->>'last_name',
        NEW.raw_user_meta_data->>'career_goal',
        NEW.raw_user_meta_data->>'experience_level'
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Trigger to create profile on user signup
DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
    AFTER INSERT ON auth.users
    FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();

-- Insert some sample data for testing (optional)
-- You can run this after creating a test user

-- Sample achievements
INSERT INTO public.achievements (user_id, title, description, category, points, icon) VALUES
    ((SELECT id FROM auth.users LIMIT 1), 'Profile Completed', 'Completed your profile setup', 'Profile', 100, '🎯'),
    ((SELECT id FROM auth.users LIMIT 1), 'First Skill Added', 'Added your first skill', 'Skills', 50, '⭐'),
    ((SELECT id FROM auth.users LIMIT 1), 'Career Goal Set', 'Set your career goal', 'Goals', 75, '🚀');

-- Sample skills
INSERT INTO public.skills (user_id, name, level, progress, category) VALUES
    ((SELECT id FROM auth.users LIMIT 1), 'JavaScript', 'Intermediate', 70, 'Programming'),
    ((SELECT id FROM auth.users LIMIT 1), 'React', 'Intermediate', 65, 'Frontend'),
    ((SELECT id FROM auth.users LIMIT 1), 'Python', 'Beginner', 30, 'Programming'),
    ((SELECT id FROM auth.users LIMIT 1), 'SQL', 'Beginner', 25, 'Database');

-- Sample activities
INSERT INTO public.activities (user_id, action, item, score) VALUES
    ((SELECT id FROM auth.users LIMIT 1), 'Completed', 'JavaScript Course', 85),
    ((SELECT id FROM auth.users LIMIT 1), 'Updated', 'Profile Information', NULL),
    ((SELECT id FROM auth.users LIMIT 1), 'Added', 'New Skill: React', NULL),
    ((SELECT id FROM auth.users LIMIT 1), 'Completed', 'Career Assessment', 92);