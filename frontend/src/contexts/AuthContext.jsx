'use client';

import React, { createContext, useContext, useEffect, useState } from 'react';

const AuthContext = createContext({});

// API configuration
const API_BASE_URL = process.env.VITE_API_BASE_URL || 'http://localhost:8000';

// API service for authentication
class AuthApiService {
  constructor() {
    this.baseURL = `${API_BASE_URL}/api/v1/auth`;
  }

  async request(endpoint, options = {}) {
    const url = `${this.baseURL}${endpoint}`;
    const config = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    };

    // Add auth token if available
    const token = localStorage.getItem('access_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }

    try {
      const response = await fetch(url, config);
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || data.message || 'Request failed');
      }

      return data;
    } catch (error) {
      console.error('API request failed:', error);
      throw error;
    }
  }

  async login(email, password) {
    return this.request('/login', {
      method: 'POST',
      body: JSON.stringify({ email, password }),
    });
  }

  async register(email, password, userData = {}) {
    const registerData = {
      email,
      password,
      full_name: userData.firstName && userData.lastName 
        ? `${userData.firstName} ${userData.lastName}` 
        : userData.full_name || null,
    };

    return this.request('/register', {
      method: 'POST',
      body: JSON.stringify(registerData),
    });
  }

  async refreshToken(refreshToken) {
    return this.request('/refresh', {
      method: 'POST',
      body: JSON.stringify({ refresh_token: refreshToken }),
    });
  }

  async getCurrentUser() {
    return this.request('/me');
  }

  async verifyToken() {
    return this.request('/verify');
  }

  async logout() {
    return this.request('/logout', {
      method: 'POST',
    });
  }
}

const authApi = new AuthApiService();

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Token management
  const getStoredTokens = () => {
    return {
      accessToken: localStorage.getItem('access_token'),
      refreshToken: localStorage.getItem('refresh_token'),
    };
  };

  const setStoredTokens = (tokens) => {
    if (tokens.access_token) {
      localStorage.setItem('access_token', tokens.access_token);
    }
    if (tokens.refresh_token) {
      localStorage.setItem('refresh_token', tokens.refresh_token);
    }
  };

  const clearStoredTokens = () => {
    localStorage.removeItem('access_token');
    localStorage.removeItem('refresh_token');
  };

  // Auto-refresh token before expiry
  const scheduleTokenRefresh = (expiresIn) => {
    // Refresh 5 minutes before expiry
    const refreshTime = (expiresIn - 300) * 1000;
    if (refreshTime > 0) {
      setTimeout(async () => {
        await refreshToken();
      }, refreshTime);
    }
  };

  // Initialize auth state
  useEffect(() => {
    const initializeAuth = async () => {
      try {
        const { accessToken, refreshToken } = getStoredTokens();
        
        if (!accessToken || !refreshToken) {
          setLoading(false);
          return;
        }

        // Verify current token
        try {
          const userData = await authApi.getCurrentUser();
          setUser(userData);
        } catch (error) {
          // Token might be expired, try to refresh
          try {
            const tokens = await authApi.refreshToken(refreshToken);
            setStoredTokens(tokens);
            scheduleTokenRefresh(tokens.expires_in);
            
            // Get user data with new token
            const userData = await authApi.getCurrentUser();
            setUser(userData);
          } catch (refreshError) {
            // Refresh failed, clear tokens
            clearStoredTokens();
            setUser(null);
          }
        }
      } catch (error) {
        console.error('Auth initialization failed:', error);
        clearStoredTokens();
        setUser(null);
      } finally {
        setLoading(false);
      }
    };

    initializeAuth();
  }, []);

  const signIn = async (email, password) => {
    try {
      setLoading(true);
      setError(null);

      const tokens = await authApi.login(email, password);
      setStoredTokens(tokens);
      scheduleTokenRefresh(tokens.expires_in);

      // Get user data
      const userData = await authApi.getCurrentUser();
      setUser(userData);

      return userData;
    } catch (error) {
      setError(error.message);
      throw error;
    } finally {
      setLoading(false);
    }
  };

  const signUp = async (email, password, userData = {}) => {
    try {
      setLoading(true);
      setError(null);

      // Register user
      const newUser = await authApi.register(email, password, userData);
      
      // Auto-login after registration
      const tokens = await authApi.login(email, password);
      setStoredTokens(tokens);
      scheduleTokenRefresh(tokens.expires_in);

      // Get updated user data
      const currentUser = await authApi.getCurrentUser();
      setUser(currentUser);

      return currentUser;
    } catch (error) {
      setError(error.message);
      throw error;
    } finally {
      setLoading(false);
    }
  };

  const signOut = async () => {
    try {
      setLoading(true);
      
      // Call logout endpoint
      try {
        await authApi.logout();
      } catch (error) {
        // Continue with logout even if API call fails
        console.warn('Logout API call failed:', error);
      }

      // Clear local state and tokens
      clearStoredTokens();
      setUser(null);
      setError(null);
    } catch (error) {
      console.error('Logout failed:', error);
      // Still clear local state on error
      clearStoredTokens();
      setUser(null);
    } finally {
      setLoading(false);
    }
  };

  const refreshToken = async () => {
    try {
      const { refreshToken: storedRefreshToken } = getStoredTokens();
      
      if (!storedRefreshToken) {
        throw new Error('No refresh token available');
      }

      const tokens = await authApi.refreshToken(storedRefreshToken);
      setStoredTokens(tokens);
      scheduleTokenRefresh(tokens.expires_in);

      return tokens;
    } catch (error) {
      console.error('Token refresh failed:', error);
      // Clear tokens and user state on refresh failure
      clearStoredTokens();
      setUser(null);
      throw error;
    }
  };

  const value = {
    user,
    loading,
    error,
    signIn,
    signUp,
    signOut,
    refreshToken,
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}

export default AuthContext;