import React, { useState } from 'react';
import { motion } from 'motion/react';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from './ui/dialog';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Separator } from './ui/separator';
import { toast } from 'sonner';
import { authUtils } from '../utils/auth';
import { Github, Linkedin, Eye, EyeOff, ArrowRight } from 'lucide-react';

interface LoginModalProps {
  open: boolean;
  onClose: () => void;
  onLogin: (isSignUp?: boolean) => void;
}

export function LoginModal({ open, onClose, onLogin }: LoginModalProps) {
  const [activeTab, setActiveTab] = useState<'login' | 'signup'>('login');
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [consent, setConsent] = useState(false);
  const [formData, setFormData] = useState({
    email: '',
    password: '',
    name: '',
    careerGoal: '',
    experience: ''
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    try {
      if (activeTab === 'signup') {
        if (!consent) {
          toast.error('You must agree to the Terms and Privacy Policy.');
          setLoading(false);
          return;
        }
        const result = await authUtils.signUp(
          formData.email,
          formData.password,
          {
            name: formData.name,
            careerGoal: formData.careerGoal,
            experience: formData.experience
          }
        );
        if (result.success) {
          toast.success('Account created! Redirecting...');
          setTimeout(() => onLogin(true), 1200);
        } else {
          toast.error(result.error || 'Sign up failed');
        }
      } else {
        const result = await authUtils.signInWithPassword(
          formData.email,
          formData.password
        );
        if (result.success) {
          toast.success('Welcome back! Redirecting...');
          setTimeout(() => onLogin(false), 1200);
        } else {
          toast.error(result.error || 'Sign in failed');
        }
      }
    } catch (error: any) {
      toast.error(error.message || 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const handleSocialLogin = async (provider: 'github' | 'linkedin' | 'google') => {
    setLoading(true);
  // Map 'linkedin' to 'linkedin_oidc' for Supabase OIDC support
  const mappedProvider = provider === 'linkedin' ? 'linkedin_oidc' : provider;
  const result = await authUtils.signInWithOAuth(mappedProvider);
    if (result.success) {
      toast.success(`Redirecting to ${provider}...`);
    } else {
      setLoading(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <div className="flex justify-center mb-4">
            <button
              className={`px-4 py-2 rounded-t-lg font-semibold ${activeTab === 'login' ? 'bg-primary text-white' : 'bg-muted text-muted-foreground'}`}
              onClick={() => setActiveTab('login')}
              type="button"
            >
              Sign In
            </button>
            <button
              className={`px-4 py-2 rounded-t-lg font-semibold ${activeTab === 'signup' ? 'bg-primary text-white' : 'bg-muted text-muted-foreground'}`}
              onClick={() => setActiveTab('signup')}
              type="button"
            >
              Sign Up
            </button>
          </div>
          <DialogTitle className="text-center text-2xl">
            {activeTab === 'signup' ? 'Create Your Account' : 'Welcome Back'}
          </DialogTitle>
          <DialogDescription className="text-center">
            {activeTab === 'signup'
              ? 'Start your career journey with CareerPilot'
              : 'Sign in to continue your career journey'
            }
          </DialogDescription>
        </DialogHeader>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
          className="space-y-6"
        >
          {/* Social Login Buttons */}
          <div className="space-y-3">
            <Button
              variant="outline"
              className="w-full flex items-center justify-center space-x-2 h-11"
              onClick={() => handleSocialLogin('google')}
              disabled={loading}
            >
              {/* Google SVG icon */}
              <svg className="w-5 h-5" viewBox="0 0 24 24"><path fill="#4285F4" d="M21.35 11.1h-9.17v2.98h5.24c-.23 1.24-1.39 3.64-5.24 3.64-3.15 0-5.72-2.61-5.72-5.82s2.57-5.82 5.72-5.82c1.8 0 3.01.77 3.7 1.43l2.53-2.46C16.44 3.99 14.56 3 12.18 3 6.99 3 2.82 7.16 2.82 12.01s4.17 9.01 9.36 9.01c5.39 0 8.96-3.77 8.96-9.09 0-.61-.07-1.21-.19-1.83z"/><path fill="#34A853" d="M3.88 7.36l2.47 1.81c.68-1.31 2.01-2.81 4.31-2.81 1.31 0 2.51.5 3.44 1.36l2.53-2.46C15.44 3.99 13.56 3 11.18 3 7.61 3 4.59 5.36 3.88 7.36z"/><path fill="#FBBC05" d="M12.18 21c2.38 0 4.26-.79 5.64-2.15l-2.7-2.21c-.93.66-2.13 1.06-3.44 1.06-2.3 0-3.63-1.5-4.31-2.81l-2.47 1.81C4.59 18.65 7.61 21 12.18 21z"/><path fill="#EA4335" d="M21.35 11.1h-9.17v2.98h5.24c-.23 1.24-1.39 3.64-5.24 3.64-3.15 0-5.72-2.61-5.72-5.82s2.57-5.82 5.72-5.82c1.8 0 3.01.77 3.7 1.43l2.53-2.46C16.44 3.99 14.56 3 12.18 3 6.99 3 2.82 7.16 2.82 12.01s4.17 9.01 9.36 9.01c5.39 0 8.96-3.77 8.96-9.09 0-.61-.07-1.21-.19-1.83z"/></svg>
              <span>Continue with Google</span>
              {loading && (
                <svg className="animate-spin h-4 w-4 ml-2 text-primary" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z" />
                </svg>
              )}
            </Button>
            <Button
              variant="outline"
              className="w-full flex items-center justify-center space-x-2 h-11"
              onClick={() => handleSocialLogin('github')}
              disabled={loading}
            >
              <Github className="w-5 h-5" />
              <span>Continue with GitHub</span>
              {loading && (
                <svg className="animate-spin h-4 w-4 ml-2 text-primary" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z" />
                </svg>
              )}
            </Button>
            <Button
              variant="outline"
              className="w-full flex items-center justify-center space-x-2 h-11"
              onClick={() => handleSocialLogin('linkedin')}
              disabled={loading}
            >
              <Linkedin className="w-5 h-5" />
              <span>Continue with LinkedIn</span>
              {loading && (
                <svg className="animate-spin h-4 w-4 ml-2 text-primary" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z" />
                </svg>
              )}
            </Button>
          </div>

          <div className="relative">
            <div className="absolute inset-0 flex items-center">
              <Separator className="w-full" />
            </div>
            <div className="relative flex justify-center text-xs uppercase">
              <span className="bg-background px-2 text-muted-foreground">
                Or continue with email
              </span>
            </div>
          </div>

          <form onSubmit={handleSubmit} className="space-y-4">
            {activeTab === 'signup' && (
              <>
                <div className="space-y-2">
                  <Label htmlFor="name">Full Name</Label>
                  <Input
                    id="name"
                    placeholder="Enter your full name"
                    value={formData.name}
                    onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                    required
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="careerGoal">Career Goal</Label>
                  <Input
                    id="careerGoal"
                    placeholder="e.g. Software Engineer, Designer, etc."
                    value={formData.careerGoal}
                    onChange={(e) => setFormData({ ...formData, careerGoal: e.target.value })}
                    required
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="experience">Experience Level</Label>
                  <select
                    id="experience"
                    className="w-full border rounded px-3 py-2"
                    value={formData.experience}
                    onChange={(e) => setFormData({ ...formData, experience: e.target.value })}
                    required
                  >
                    <option value="">Select experience</option>
                    <option value="student">Student</option>
                    <option value="entry">Entry Level</option>
                    <option value="mid">Mid Level</option>
                    <option value="senior">Senior</option>
                  </select>
                </div>
              </>
            )}

            <div className="space-y-2">
              <Label htmlFor="email">Email</Label>
              <Input
                id="email"
                type="email"
                placeholder="Enter your email"
                value={formData.email}
                onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                required
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="password">Password</Label>
              <div className="relative">
                <Input
                  id="password"
                  type={showPassword ? 'text' : 'password'}
                  placeholder="Enter your password"
                  value={formData.password}
                  onChange={(e) => setFormData({ ...formData, password: e.target.value })}
                  required
                />
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  className="absolute right-0 top-0 h-full px-3 py-2 hover:bg-transparent"
                  onClick={() => setShowPassword(!showPassword)}
                >
                  {showPassword ? (
                    <EyeOff className="h-4 w-4" />
                  ) : (
                    <Eye className="h-4 w-4" />
                  )}
                </Button>
              </div>
            </div>

            {activeTab === 'signup' && (
              <div className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  id="consent"
                  checked={consent}
                  onChange={e => setConsent(e.target.checked)}
                  className="accent-primary"
                  required
                />
                <label htmlFor="consent" className="text-xs text-muted-foreground">
                  I agree to the <a href="#" className="underline hover:text-primary">Terms of Service</a> and <a href="#" className="underline hover:text-primary">Privacy Policy</a>
                </label>
              </div>
            )}

            <Button type="submit" className="w-full h-11" disabled={loading}>
              {loading ? (
                <span className="flex items-center justify-center">
                  <svg className="animate-spin h-5 w-5 mr-2 text-primary" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z" />
                  </svg>
                  Loading...
                </span>
              ) : activeTab === 'signup' ? 'Create Account' : 'Sign In'}
              {!loading && <ArrowRight className="w-4 h-4 ml-2" />}
            </Button>
          </form>
        </motion.div>
      </DialogContent>
    </Dialog>
  );
}