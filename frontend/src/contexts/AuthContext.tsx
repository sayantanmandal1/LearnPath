'use client';

import React, { createContext, useContext, useEffect, useState } from 'react';
import { supabase } from '../utils/supabase/client';
import type { User as SupabaseUser, Session } from '@supabase/supabase-js';

// Error handling utility for production
const getErrorMessage = (error: unknown): string => {
  if (error instanceof Error) return error.message;
  if (typeof error === 'string') return error;
  if (error && typeof error === 'object' && 'message' in error) {
    return String(error.message);
  }
  return 'An unexpected error occurred';
};

// Types
interface User {
  id: string;
  email: string;
  full_name?: string;
  user_metadata?: any;
  [key: string]: any;
}

interface AuthContextType {
  user: User | null;
  session: Session | null;
  loading: boolean;
  error: string | null;
  signIn: (email: string, password: string) => Promise<User>;
  signUp: (email: string, password: string, userData?: any) => Promise<User>;
  signOut: () => Promise<void>;
  signInWithOAuth: (provider: 'google' | 'github' | 'linkedin_oidc') => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

interface AuthProviderProps {
  children: React.ReactNode;
}

export function AuthProvider({ children }: AuthProviderProps) {
  const [user, setUser] = useState<User | null>(null);
  const [session, setSession] = useState<Session | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Initialize auth state
  useEffect(() => {
    const initializeAuth = async () => {
      try {
        // Get initial session
        const { data: { session }, error } = await supabase.auth.getSession();
        
        if (error) {
          console.error('Error getting session:', error);
          setError(getErrorMessage(error));
        } else {
          setSession(session);
          setUser(session?.user ? {
            id: session.user.id,
            email: session.user.email || '',
            full_name: session.user.user_metadata?.full_name || session.user.user_metadata?.name,
            user_metadata: session.user.user_metadata
          } : null);
        }
      } catch (error) {
        console.error('Auth initialization failed:', getErrorMessage(error));
        setError(getErrorMessage(error));
      } finally {
        setLoading(false);
      }
    };

    initializeAuth();

    // Listen for auth changes
    const { data: { subscription } } = supabase.auth.onAuthStateChange(
      async (event, session) => {
        console.log('Auth state changed:', event, session);
        setSession(session);
        setUser(session?.user ? {
          id: session.user.id,
          email: session.user.email || '',
          full_name: session.user.user_metadata?.full_name || session.user.user_metadata?.name,
          user_metadata: session.user.user_metadata
        } : null);
        setLoading(false);
        setError(null);
      }
    );

    return () => subscription.unsubscribe();
  }, []);

  const signIn = async (email: string, password: string): Promise<User> => {
    try {
      setLoading(true);
      setError(null);

      const { data, error } = await supabase.auth.signInWithPassword({
        email,
        password,
      });

      if (error) {
        throw error;
      }

      if (!data.user) {
        throw new Error('No user returned from sign in');
      }

      const userData: User = {
        id: data.user.id,
        email: data.user.email || '',
        full_name: data.user.user_metadata?.full_name || data.user.user_metadata?.name,
        user_metadata: data.user.user_metadata
      };

      return userData;
    } catch (error) {
      const errorMessage = getErrorMessage(error);
      setError(errorMessage);
      throw error;
    } finally {
      setLoading(false);
    }
  };

  const signUp = async (email: string, password: string, userData: any = {}): Promise<User> => {
    try {
      setLoading(true);
      setError(null);

      const { data, error } = await supabase.auth.signUp({
        email,
        password,
        options: {
          data: {
            full_name: userData.firstName && userData.lastName
              ? `${userData.firstName} ${userData.lastName}`
              : userData.full_name || '',
            first_name: userData.firstName || '',
            last_name: userData.lastName || '',
            career_goal: userData.careerGoal || '',
            experience_level: userData.experience || ''
          }
        }
      });

      if (error) {
        throw error;
      }

      if (!data.user) {
        throw new Error('No user returned from sign up');
      }

      const newUser: User = {
        id: data.user.id,
        email: data.user.email || '',
        full_name: data.user.user_metadata?.full_name || data.user.user_metadata?.name,
        user_metadata: data.user.user_metadata
      };

      return newUser;
    } catch (error) {
      const errorMessage = getErrorMessage(error);
      setError(errorMessage);
      throw error;
    } finally {
      setLoading(false);
    }
  };

  const signOut = async (): Promise<void> => {
    try {
      setLoading(true);
      const { error } = await supabase.auth.signOut();
      
      if (error) {
        throw error;
      }

      setUser(null);
      setSession(null);
      setError(null);
    } catch (error) {
      console.error('Logout failed:', getErrorMessage(error));
      setError(getErrorMessage(error));
    } finally {
      setLoading(false);
    }
  };

  const signInWithOAuth = async (provider: 'google' | 'github' | 'linkedin_oidc'): Promise<void> => {
    try {
      setLoading(true);
      setError(null);

      const { data, error } = await supabase.auth.signInWithOAuth({
        provider,
        options: {
          redirectTo: `${window.location.origin}/dashboard`
        }
      });

      if (error) {
        throw error;
      }
    } catch (error) {
      const errorMessage = getErrorMessage(error);
      setError(errorMessage);
      throw error;
    } finally {
      setLoading(false);
    }
  };

  const value = {
    user,
    session,
    loading,
    error,
    signIn,
    signUp,
    signOut,
    signInWithOAuth,
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth(): AuthContextType {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}

export default AuthContext;