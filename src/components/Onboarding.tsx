import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { supabase } from '../utils/supabase/client';
import { apiPost } from '../api';
import { motion } from 'motion/react';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { toast } from 'sonner';

export function Onboarding() {
  const navigate = useNavigate();
  const [form, setForm] = useState({
    name: '',
    careerGoal: '',
    experience: '',
  });
  const [loading, setLoading] = useState(false);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    try {
      // Get current user from Supabase (for ID/email)
      const { data: { user }, error: userError } = await supabase.auth.getUser();
      if (userError || !user) {
        toast.error('Could not get user. Please log in again.');
        setLoading(false);
        return;
      }
      // Send profile data to backend API
      await apiPost(`/profile`, {
        user_id: user.id,
        name: form.name,
        career_goal: form.careerGoal,
        experience: form.experience,
        email: user.email,
      });
      toast.success('Profile created!');
      navigate('/dashboard');
    } catch (err: any) {
      toast.error('Unexpected error: ' + (err.message || err));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex items-center justify-center min-h-[60vh]">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="w-full max-w-md p-8 bg-white rounded-2xl shadow-2xl border border-border flex flex-col items-center"
      >
        <h1 className="text-3xl font-bold mb-4 text-center">Welcome to CareerPilot!</h1>
        <p className="text-lg text-muted-foreground mb-6 text-center">
          Let&apos;s set up your profile to personalize your experience.
        </p>
        <form className="space-y-6 w-full" onSubmit={handleSubmit}>
          <div className="space-y-2">
            <Label htmlFor="name">Full Name</Label>
            <Input
              id="name"
              name="name"
              placeholder="Enter your full name"
              value={form.name}
              onChange={handleChange}
              required
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="careerGoal">Career Goal</Label>
            <Input
              id="careerGoal"
              name="careerGoal"
              placeholder="e.g. Software Engineer, Designer, etc."
              value={form.careerGoal}
              onChange={handleChange}
              required
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="experience">Experience Level</Label>
            <select
              id="experience"
              name="experience"
              className="w-full border rounded px-3 py-2"
              value={form.experience}
              onChange={handleChange}
              required
            >
              <option value="">Select experience</option>
              <option value="student">Student</option>
              <option value="entry">Entry Level</option>
              <option value="mid">Mid Level</option>
              <option value="senior">Senior</option>
            </select>
          </div>
          <Button type="submit" className="w-full h-12" disabled={loading}>
            {loading ? 'Saving...' : 'Complete Onboarding'}
          </Button>
        </form>
      </motion.div>
    </div>
  );
}
