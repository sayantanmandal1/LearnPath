'use client';

import React, { createContext, useContext, useEffect, useState } from 'react';
import toast from 'react-hot-toast';

const AuthContext = createContext({});

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [userProfile, setUserProfile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [initializing, setInitializing] = useState(false);

  // Mock authentication functions for now
  const signUp = async (email, password, additionalData = {}) => {
    try {
      setLoading(true);
      // Mock user creation
      const mockUser = {
        uid: 'mock-uid-' + Date.now(),
        email,
        displayName: additionalData.displayName || '',
        emailVerified: false
      };
      
      setUser(mockUser);
      toast.success('Account created! (Mock mode)');
      return { user: mockUser, success: true };
    } catch (error) {
      toast.error('Sign up failed');
      return { error, success: false };
    } finally {
      setLoading(false);
    }
  };

  const signIn = async (email, password) => {
    try {
      setLoading(true);
      // Mock sign in
      const mockUser = {
        uid: 'mock-uid-signin',
        email,
        displayName: 'Mock User',
        emailVerified: true
      };
      
      setUser(mockUser);
      toast.success('Welcome back! (Mock mode)');
      return { user: mockUser, success: true };
    } catch (error) {
      toast.error('Sign in failed');
      return { error, success: false };
    } finally {
      setLoading(false);
    }
  };

  const signInWithGoogle = async () => {
    try {
      setLoading(true);
      const mockUser = {
        uid: 'mock-google-uid',
        email: 'user@gmail.com',
        displayName: 'Google User',
        emailVerified: true
      };
      
      setUser(mockUser);
      toast.success('Signed in with Google! (Mock mode)');
      return { user: mockUser, success: true };
    } catch (error) {
      toast.error('Google sign in failed');
      return { error, success: false };
    } finally {
      setLoading(false);
    }
  };

  const signInWithGitHub = async () => {
    try {
      setLoading(true);
      const mockUser = {
        uid: 'mock-github-uid',
        email: 'user@github.com',
        displayName: 'GitHub User',
        emailVerified: true
      };
      
      setUser(mockUser);
      toast.success('Signed in with GitHub! (Mock mode)');
      return { user: mockUser, success: true };
    } catch (error) {
      toast.error('GitHub sign in failed');
      return { error, success: false };
    } finally {
      setLoading(false);
    }
  };

  const signInWithTwitter = async () => {
    try {
      setLoading(true);
      const mockUser = {
        uid: 'mock-twitter-uid',
        email: 'user@twitter.com',
        displayName: 'Twitter User',
        emailVerified: true
      };
      
      setUser(mockUser);
      toast.success('Signed in with Twitter! (Mock mode)');
      return { user: mockUser, success: true };
    } catch (error) {
      toast.error('Twitter sign in failed');
      return { error, success: false };
    } finally {
      setLoading(false);
    }
  };

  const logout = async () => {
    try {
      setUser(null);
      setUserProfile(null);
      toast.success('Signed out successfully!');
    } catch (error) {
      toast.error('Failed to sign out');
    }
  };

  const resetPassword = async (email) => {
    try {
      toast.success('Password reset email sent! (Mock mode)');
      return { success: true };
    } catch (error) {
      toast.error('Password reset failed');
      return { error, success: false };
    }
  };

  const changePassword = async (currentPassword, newPassword) => {
    try {
      toast.success('Password updated successfully! (Mock mode)');
      return { success: true };
    } catch (error) {
      toast.error('Password update failed');
      return { error, success: false };
    }
  };

  const updateUserProfile = async (uid, updates) => {
    try {
      setUserProfile(prev => ({ ...prev, ...updates }));
      toast.success('Profile updated successfully! (Mock mode)');
      return true;
    } catch (error) {
      toast.error('Failed to update profile');
      return false;
    }
  };

  const updateAnalytics = async (analyticsData) => {
    // Mock analytics update
    console.log('Analytics updated:', analyticsData);
  };

  const checkUsernameAvailability = async (username) => {
    // Mock username check
    return username !== 'taken';
  };

  const createUserProfile = async (user, additionalData = {}) => {
    // Mock profile creation
    return true;
  };

  const getUserProfile = async (uid) => {
    // Mock profile fetch
    return null;
  };

  const value = {
    user,
    userProfile,
    loading,
    initializing,
    signUp,
    signIn,
    signInWithGoogle,
    signInWithGitHub,
    signInWithTwitter,
    logout,
    resetPassword,
    changePassword,
    updateUserProfile,
    updateAnalytics,
    checkUsernameAvailability,
    createUserProfile,
    getUserProfile
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};