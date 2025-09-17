import { supabase } from './supabase/client';
import { toast } from 'sonner';

export const authUtils = {
  /**
   * Sign out the current user
   */
  signOut: async () => {
    try {
      const { error } = await supabase.auth.signOut();
      if (error) throw error;
      
      toast.success('Signed out successfully');
      return { success: true };
    } catch (error: any) {
      toast.error('Error signing out: ' + error.message);
      return { success: false, error: error.message };
    }
  },

  /**
   * Get the current session
   */
  getSession: async () => {
    try {
      const { data: { session }, error } = await supabase.auth.getSession();
      if (error) throw error;
      
      return { session, user: session?.user || null };
    } catch (error: any) {
      console.error('Error getting session:', error);
      return { session: null, user: null };
    }
  },

  /**
   * Get the current user
   */
  getUser: async () => {
    try {
      const { data: { user }, error } = await supabase.auth.getUser();
      if (error) throw error;
      
      return { user };
    } catch (error: any) {
      console.error('Error getting user:', error);
      return { user: null };
    }
  },

  /**
   * Sign in with email and password
   */
  signInWithPassword: async (email: string, password: string) => {
    try {
      const { data, error } = await supabase.auth.signInWithPassword({
        email,
        password,
      });
      
      if (error) throw error;
      
      toast.success('Welcome back!');
      return { success: true, user: data.user, session: data.session };
    } catch (error: any) {
      toast.error('Sign in failed: ' + error.message);
      return { success: false, error: error.message };
    }
  },

  /**
   * Sign up with email and password
   */
  signUp: async (email: string, password: string, metadata?: any) => {
    try {
      const { data, error } = await supabase.auth.signUp({
        email,
        password,
        options: {
          data: metadata,
        },
      });
      
      if (error) throw error;
      
      toast.success('Account created! Check your email for verification.');
      return { success: true, user: data.user, session: data.session };
    } catch (error: any) {
      let msg = error.message || '';
      if (msg.toLowerCase().includes('already registered') || msg.toLowerCase().includes('user already exists')) {
        toast.error('This account already exists.');
        return { success: false, error: 'This account already exists.' };
      } else {
        toast.error('Invalid credentials.');
        return { success: false, error: 'Invalid credentials.' };
      }
    }
  },

  /**
   * Sign in with OAuth (GitHub, LinkedIn, etc.)
   */
  signInWithOAuth: async (provider: 'github' | 'linkedin' | 'linkedin_oidc' | 'google') => {
    try {
      const { data, error } = await supabase.auth.signInWithOAuth({
        provider,
        options: {
          redirectTo: window.location.origin,
        },
      });
      
      if (error) throw error;
      
      return { success: true, data };
    } catch (error: any) {
      toast.error(`${provider} sign in failed: ` + error.message);
      return { success: false, error: error.message };
    }
  }
};