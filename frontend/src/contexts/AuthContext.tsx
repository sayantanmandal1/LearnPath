'use client';

import React, { createContext, useContext, useEffect, useState } from 'react';

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
  [key: string]: any;
}

interface AuthTokens {
  access_token: string;
  refresh_token: string;
  expires_in?: number;
}

interface AuthContextType {
  user: User | null;
  loading: boolean;
  error: string | null;
  signIn: (email: string, password: string) => Promise<User>;
  signUp: (email: string, password: string, userData?: any) => Promise<User>;
  signOut: () => Promise<void>;
  refreshToken: () => Promise<AuthTokens>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

// API configuration
const API_BASE_URL = process.env.VITE_API_BASE_URL || 'http://localhost:8000';

// API service for authentication
class AuthApiService {
  private baseURL: string;

  constructor() {
    this.baseURL = `${API_BASE_URL}/api/v1/auth`;
  }

  async request(endpoint: string, options: RequestInit = {}): Promise<any> {
    const url = `${this.baseURL}${endpoint}`;
    const config: RequestInit = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    };

    // Add auth token if available
    const token = localStorage.getItem('access_token');
    if (token) {
      (config.headers as Record<string, string>).Authorization = `Bearer ${token}`;
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

  async login(email: string, password: string) {
    return this.request('/login', {
      method: 'POST',
      body: JSON.stringify({ email, password }),
    });
  }

  async register(email: string, password: string, userData: any = {}) {
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

  async refreshToken(refreshToken: string) {
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

interface AuthProviderProps {
  children: React.ReactNode;
}

export function AuthProvider({ children }: AuthProviderProps) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Token management
  const getStoredTokens = () => {
    return {
      accessToken: localStorage.getItem('access_token'),
      refreshToken: localStorage.getItem('refresh_token'),
    };
  };

  const setStoredTokens = (tokens: Partial<AuthTokens>) => {
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
  const scheduleTokenRefresh = (expiresIn?: number) => {
    if (!expiresIn) return;

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
            console.warn('Token refresh failed during initialization:', getErrorMessage(refreshError));
            clearStoredTokens();
            setUser(null);
          }
        }
      } catch (error) {
        console.error('Auth initialization failed:', getErrorMessage(error));
        clearStoredTokens();
        setUser(null);
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

      const tokens = await authApi.login(email, password);
      setStoredTokens(tokens);
      scheduleTokenRefresh(tokens.expires_in);

      // Get user data
      const userData = await authApi.getCurrentUser();
      setUser(userData);

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

      // Call logout endpoint
      try {
        await authApi.logout();
      } catch (error) {
        // Continue with logout even if API call fails
        console.warn('Logout API call failed:', getErrorMessage(error));
      }

      // Clear local state and tokens
      clearStoredTokens();
      setUser(null);
      setError(null);
    } catch (error) {
      console.error('Logout failed:', getErrorMessage(error));
      // Still clear local state on error
      clearStoredTokens();
      setUser(null);
    } finally {
      setLoading(false);
    }
  };

  const refreshToken = async (): Promise<AuthTokens> => {
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
      console.error('Token refresh failed:', getErrorMessage(error));
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

export function useAuth(): AuthContextType {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}

export default AuthContext;