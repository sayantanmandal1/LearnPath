'use client';

import React, { createContext, useContext, useEffect, useState } from 'react';
import { supabase } from '../utils/supabase/client';
import type { Session } from '@supabase/supabase-js';

// Error handling utility for production
const getErrorMessage = (error: unknown): string => {
  if (error instanceof Error) return error.message;
  if (typeof error === 'string') return error;
  if (error && typeof error === 'object' && 'message' in error) {
    return String(error.message);
  }
  return 'An unexpected error occurred';
};

// Types - Updated to match backend response
interface User {
  id: string;
  email: string;
  full_name?: string;
  is_active: boolean;
  is_verified: boolean;
  created_at: string;
  last_login?: string;
  profile?: {
    id: string;
    dream_job?: string;
    experience_years?: number;
    current_role?: string;
    location?: string;
    github_username?: string;
    leetcode_id?: string;
    linkedin_url?: string;
    skills?: string[];
    created_at?: string;
    updated_at?: string;
  };
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
  refreshToken: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

interface AuthProviderProps {
  children: React.ReactNode;
}

// Backend API base URL
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export function AuthProvider({ children }: AuthProviderProps) {
  const [user, setUser] = useState<User | null>(null);
  const [session, setSession] = useState<Session | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Helper function to refresh tokens
  const refreshTokens = async (): Promise<void> => {
    const refreshToken = localStorage.getItem('refresh_token');
    if (!refreshToken) {
      throw new Error('No refresh token available');
    }

    const response = await fetch(`${API_BASE_URL}/api/v1/auth/refresh`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ refresh_token: refreshToken }),
    });

    if (!response.ok) {
      localStorage.removeItem('access_token');
      localStorage.removeItem('refresh_token');
      setUser(null);
      throw new Error('Failed to refresh token');
    }

    const data = await response.json();
    localStorage.setItem('access_token', data.access_token);
    localStorage.setItem('refresh_token', data.refresh_token);
    setUser(data.user);
  };

  // Initialize auth state
  useEffect(() => {
    const initializeAuth = async () => {
      try {
        const accessToken = localStorage.getItem('access_token');

        if (accessToken) {
          // Verify token with backend and get user data
          try {
            const response = await fetch(`${API_BASE_URL}/api/v1/auth/me`, {
              headers: {
                'Authorization': `Bearer ${accessToken}`,
                'Content-Type': 'application/json',
              },
            });

            if (response.ok) {
              const userData = await response.json();
              setUser(userData);
            } else {
              // Invalid token, clear storage
              console.log('Invalid token, clearing storage');
              localStorage.removeItem('access_token');
              localStorage.removeItem('refresh_token');
            }
          } catch (error) {
            console.error('Error verifying token:', error);
            localStorage.removeItem('access_token');
            localStorage.removeItem('refresh_token');
          }
        }

      } catch (error) {
        console.error('Auth initialization failed:', getErrorMessage(error));
        setError(getErrorMessage(error));
      } finally {
        setLoading(false);
      }
    };

    initializeAuth();
  }, []);

  const signIn = async (email: string, password: string): Promise<User> => {
    try {
      setLoading(true);
      setError(null);

      // Use backend API for authentication
      const response = await fetch(`${API_BASE_URL}/api/v1/auth/login`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ email, password }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Login failed');
      }

      const data = await response.json();

      // Store tokens
      localStorage.setItem('access_token', data.access_token);
      localStorage.setItem('refresh_token', data.refresh_token);

      // Set user data
      setUser(data.user);

      return data.user;
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

      // Use backend API for registration
      const response = await fetch(`${API_BASE_URL}/api/v1/auth/register`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          email,
          password,
          full_name: userData.firstName && userData.lastName
            ? `${userData.firstName} ${userData.lastName}`
            : userData.full_name || '',
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Registration failed');
      }

      const data = await response.json();

      // After successful registration, automatically sign in
      return await signIn(email, password);
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
      setError(null);

      // Sign out from Supabase only if there's a session (OAuth)
      if (session) {
        try {
          await supabase.auth.signOut();
        } catch (supabaseError) {
          console.log('Supabase signout failed, continuing with backend signout');
        }
      }

      // Clear backend tokens
      localStorage.removeItem('access_token');
      localStorage.removeItem('refresh_token');

      // Clear user state
      setUser(null);
      setSession(null);
    } catch (error) {
      const errorMessage = getErrorMessage(error);
      setError(errorMessage);
      throw error;
    } finally {
      setLoading(false);
    }
  };

  const signInWithOAuth = async (provider: 'google' | 'github' | 'linkedin_oidc'): Promise<void> => {
    try {
      setLoading(true);
      setError(null);

      // Use Supabase for OAuth
      const { error } = await supabase.auth.signInWithOAuth({
        provider,
        options: {
          redirectTo: `${window.location.origin}/auth/callback`,
        },
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

  const refreshToken = async (): Promise<void> => {
    await refreshTokens();
  };

  // Helper method to clear all auth data
  const clearAllAuthData = () => {
    localStorage.removeItem('access_token');
    localStorage.removeItem('refresh_token');
    // Clear Supabase session data
    localStorage.removeItem('sb-bmhvwzqadllsyncnyhyw-auth-token');
    sessionStorage.clear();
    setUser(null);
    setSession(null);
    setError(null);
  };

  const value: AuthContextType = {
    user,
    session,
    loading,
    error,
    signIn,
    signUp,
    signOut,
    signInWithOAuth,
    refreshToken,
  };

  // Clear all auth data on mount to avoid conflicts
  useEffect(() => {
    // Clear any stale Supabase sessions that might be causing issues
    const clearStaleData = () => {
      const supabaseKeys = Object.keys(localStorage).filter(key =>
        key.startsWith('sb-') || key.includes('supabase')
      );
      supabaseKeys.forEach(key => localStorage.removeItem(key));
    };

    clearStaleData();
  }, []);

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