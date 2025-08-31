'use client';

import React, { createContext, useContext, useEffect, useState } from 'react';
import {
  signInWithEmailAndPassword,
  createUserWithEmailAndPassword,
  signInWithPopup,
  signOut,
  onAuthStateChanged,
  updateProfile,
  sendPasswordResetEmail,
  sendEmailVerification,
  updatePassword,
  reauthenticateWithCredential,
  EmailAuthProvider
} from 'firebase/auth';
import {
  doc,
  setDoc,
  getDoc,
  updateDoc,
  serverTimestamp,
  collection,
  query,
  where,
  getDocs
} from 'firebase/firestore';
import { auth, db, googleProvider, githubProvider, twitterProvider } from '../lib/firebase';
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
  const [loading, setLoading] = useState(true);
  const [initializing, setInitializing] = useState(true);

  // Create or update user profile in Firestore
  const createUserProfile = async (user, additionalData = {}) => {
    if (!user) return;

    const userRef = doc(db, 'users', user.uid);
    const userSnap = await getDoc(userRef);

    if (!userSnap.exists()) {
      const { displayName, email, photoURL, uid } = user;
      const createdAt = serverTimestamp();

      try {
        await setDoc(userRef, {
          uid,
          displayName: displayName || additionalData.displayName || '',
          email,
          photoURL: photoURL || '',
          createdAt,
          updatedAt: createdAt,
          isActive: true,
          preferences: {
            theme: 'dark',
            notifications: {
              email: true,
              push: true,
              jobAlerts: true,
              careerUpdates: true
            },
            privacy: {
              profileVisibility: 'public',
              showEmail: false,
              showPhone: false
            }
          },
          profile: {
            firstName: additionalData.firstName || '',
            lastName: additionalData.lastName || '',
            title: additionalData.title || '',
            bio: additionalData.bio || '',
            location: additionalData.location || '',
            website: additionalData.website || '',
            linkedin: additionalData.linkedin || '',
            github: additionalData.github || '',
            twitter: additionalData.twitter || '',
            phone: additionalData.phone || '',
            skills: additionalData.skills || [],
            experience: additionalData.experience || [],
            education: additionalData.education || [],
            certifications: additionalData.certifications || [],
            languages: additionalData.languages || [],
            interests: additionalData.interests || [],
            careerGoals: additionalData.careerGoals || '',
            salaryExpectation: additionalData.salaryExpectation || null,
            availableForWork: additionalData.availableForWork || false,
            preferredJobTypes: additionalData.preferredJobTypes || [],
            preferredLocations: additionalData.preferredLocations || [],
            remoteWork: additionalData.remoteWork || false
          },
          analytics: {
            profileViews: 0,
            jobApplications: 0,
            recommendationsGenerated: 0,
            skillAssessments: 0,
            lastActive: serverTimestamp()
          },
          ...additionalData
        });

        toast.success('Profile created successfully!');
      } catch (error) {
        console.error('Error creating user profile:', error);
        toast.error('Failed to create profile');
      }
    }

    return userRef;
  };

  // Get user profile from Firestore
  const getUserProfile = async (uid) => {
    if (!uid) return null;

    try {
      const userRef = doc(db, 'users', uid);
      const userSnap = await getDoc(userRef);

      if (userSnap.exists()) {
        const profileData = userSnap.data();
        setUserProfile(profileData);
        return profileData;
      }
    } catch (error) {
      console.error('Error fetching user profile:', error);
    }

    return null;
  };

  // Update user profile
  const updateUserProfile = async (uid, updates) => {
    if (!uid) return false;

    try {
      const userRef = doc(db, 'users', uid);
      await updateDoc(userRef, {
        ...updates,
        updatedAt: serverTimestamp()
      });

      // Update local state
      setUserProfile(prev => ({ ...prev, ...updates }));
      toast.success('Profile updated successfully!');
      return true;
    } catch (error) {
      console.error('Error updating user profile:', error);
      toast.error('Failed to update profile');
      return false;
    }
  };

  // Sign up with email and password
  const signUp = async (email, password, additionalData = {}) => {
    try {
      setLoading(true);
      const { user } = await createUserWithEmailAndPassword(auth, email, password);
      
      // Update display name if provided
      if (additionalData.displayName) {
        await updateProfile(user, {
          displayName: additionalData.displayName
        });
      }

      // Send email verification
      await sendEmailVerification(user);
      
      // Create user profile
      await createUserProfile(user, additionalData);
      
      toast.success('Account created! Please check your email for verification.');
      return { user, success: true };
    } catch (error) {
      console.error('Sign up error:', error);
      toast.error(error.message);
      return { error, success: false };
    } finally {
      setLoading(false);
    }
  };

  // Sign in with email and password
  const signIn = async (email, password) => {
    try {
      setLoading(true);
      const { user } = await signInWithEmailAndPassword(auth, email, password);
      
      // Update last active
      await updateUserProfile(user.uid, {
        'analytics.lastActive': serverTimestamp()
      });
      
      toast.success('Welcome back!');
      return { user, success: true };
    } catch (error) {
      console.error('Sign in error:', error);
      toast.error(error.message);
      return { error, success: false };
    } finally {
      setLoading(false);
    }
  };

  // Sign in with Google
  const signInWithGoogle = async () => {
    try {
      setLoading(true);
      const { user } = await signInWithPopup(auth, googleProvider);
      await createUserProfile(user);
      
      toast.success('Signed in with Google!');
      return { user, success: true };
    } catch (error) {
      console.error('Google sign in error:', error);
      toast.error(error.message);
      return { error, success: false };
    } finally {
      setLoading(false);
    }
  };

  // Sign in with GitHub
  const signInWithGitHub = async () => {
    try {
      setLoading(true);
      const { user } = await signInWithPopup(auth, githubProvider);
      await createUserProfile(user);
      
      toast.success('Signed in with GitHub!');
      return { user, success: true };
    } catch (error) {
      console.error('GitHub sign in error:', error);
      toast.error(error.message);
      return { error, success: false };
    } finally {
      setLoading(false);
    }
  };

  // Sign in with Twitter
  const signInWithTwitter = async () => {
    try {
      setLoading(true);
      const { user } = await signInWithPopup(auth, twitterProvider);
      await createUserProfile(user);
      
      toast.success('Signed in with Twitter!');
      return { user, success: true };
    } catch (error) {
      console.error('Twitter sign in error:', error);
      toast.error(error.message);
      return { error, success: false };
    } finally {
      setLoading(false);
    }
  };

  // Sign out
  const logout = async () => {
    try {
      await signOut(auth);
      setUser(null);
      setUserProfile(null);
      toast.success('Signed out successfully!');
    } catch (error) {
      console.error('Sign out error:', error);
      toast.error('Failed to sign out');
    }
  };

  // Reset password
  const resetPassword = async (email) => {
    try {
      await sendPasswordResetEmail(auth, email);
      toast.success('Password reset email sent!');
      return { success: true };
    } catch (error) {
      console.error('Password reset error:', error);
      toast.error(error.message);
      return { error, success: false };
    }
  };

  // Update password
  const changePassword = async (currentPassword, newPassword) => {
    try {
      if (!user) throw new Error('No user logged in');

      // Re-authenticate user
      const credential = EmailAuthProvider.credential(user.email, currentPassword);
      await reauthenticateWithCredential(user, credential);
      
      // Update password
      await updatePassword(user, newPassword);
      
      toast.success('Password updated successfully!');
      return { success: true };
    } catch (error) {
      console.error('Password update error:', error);
      toast.error(error.message);
      return { error, success: false };
    }
  };

  // Update user analytics
  const updateAnalytics = async (analyticsData) => {
    if (!user) return;

    try {
      const userRef = doc(db, 'users', user.uid);
      await updateDoc(userRef, {
        [`analytics.${Object.keys(analyticsData)[0]}`]: Object.values(analyticsData)[0],
        'analytics.lastActive': serverTimestamp()
      });
    } catch (error) {
      console.error('Error updating analytics:', error);
    }
  };

  // Check if username is available
  const checkUsernameAvailability = async (username) => {
    try {
      const usersRef = collection(db, 'users');
      const q = query(usersRef, where('profile.username', '==', username));
      const querySnapshot = await getDocs(q);
      
      return querySnapshot.empty;
    } catch (error) {
      console.error('Error checking username:', error);
      return false;
    }
  };

  // Monitor auth state changes
  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, async (user) => {
      if (user) {
        setUser(user);
        await getUserProfile(user.uid);
      } else {
        setUser(null);
        setUserProfile(null);
      }
      
      if (initializing) {
        setInitializing(false);
      }
      setLoading(false);
    });

    return unsubscribe;
  }, [initializing]);

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