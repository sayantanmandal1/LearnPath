import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { motion, AnimatePresence } from 'motion/react';
import { Navigation } from './components/Navigation';
import { Link } from 'react-router-dom';
import { LandingPage } from './components/LandingPage';
import { Dashboard } from './components/Dashboard';
import { Onboarding } from './components/Onboarding';
import { Profile } from './components/Profile';
import { Analysis } from './components/Analysis';
import { About } from './components/About';
import { Features } from './components/Features';
import { FAQ } from './components/FAQ';
import { NotFound } from './components/NotFound';
import { LoadingSpinner } from './components/LoadingSpinner';
import { LoginModal } from './components/LoginModal';
import { ErrorBoundary } from './components/ErrorBoundary';
import { Toaster } from './components/ui/sonner';
import { AuthProvider, useAuth } from './contexts/AuthContext';

function AppContent() {
  const { user, loading, signOut } = useAuth();
  const [showLoginModal, setShowLoginModal] = useState(false);
  const [profileChecked, setProfileChecked] = useState(false);
  const [needsOnboarding, setNeedsOnboarding] = useState(false);

  // Check if user profile is missing and needs onboarding
  useEffect(() => {
    const checkProfile = async () => {
      if (!user) {
        setProfileChecked(true);
        setNeedsOnboarding(false);
        return;
      }

      // TODO: Replace with backend API call to check user profile
      // For now, assume new users need onboarding
      try {
        // This would be replaced with a call to the backend profile API
        // const response = await fetch(`${API_BASE_URL}/api/v1/profiles/me`);
        // const profileData = await response.json();

        // For now, check if user has full_name (indicating completed registration)
        if (!user.full_name) {
          setNeedsOnboarding(true);
        } else {
          setNeedsOnboarding(false);
        }
      } catch (error) {
        console.error('Error checking profile:', error);
        setNeedsOnboarding(false);
      }
      setProfileChecked(true);
    };

    if (user) {
      checkProfile();
    } else {
      setProfileChecked(true);
      setNeedsOnboarding(false);
    }
  }, [user]);

  const handleSignOut = async () => {
    if (!window.confirm('Are you sure you want to sign out?')) return;
    try {
      await signOut();
      window.location.replace('/');
    } catch (error) {
      console.error('Error signing out:', error);
    }
  };

  const handleLoginSuccess = (isSignUp = false) => {
    setShowLoginModal(false);
    if (isSignUp) {
      window.location.replace('/onboarding');
    } else {
      window.location.replace('/dashboard');
    }
  };

  const isLoggedIn = !!user;

  // Show loading spinner while checking authentication or profile
  if (loading || !profileChecked) {
    return <LoadingSpinner />;
  }

  // Only require onboarding for protected routes
  const protectedRoutes = ['/dashboard', '/profile', '/analysis'];
  if (needsOnboarding && protectedRoutes.includes(window.location.pathname)) {
    if (window.location.pathname !== '/onboarding') {
      window.location.replace('/onboarding');
      return null;
    }
    // Show only logo and onboarding card
    return (
      <div className="min-h-screen flex flex-col items-center justify-center bg-background">
        <div className="w-full flex justify-center pt-8">
          <Link to="/" className="flex items-center space-x-2 select-none">
            <div className="w-8 h-8 bg-primary rounded-lg flex items-center justify-center">
              <svg className="w-5 h-5 text-primary-foreground" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path d="M2.5 19.5L21.5 12L2.5 4.5V10.5L17.5 12L2.5 13.5V19.5Z" /></svg>
            </div>
            <span className="text-xl font-semibold">CareerPilot</span>
          </Link>
        </div>
        <div className="flex-1 flex items-center justify-center w-full">
          <Onboarding />
        </div>
      </div>
    );
  }

  return (
    <ErrorBoundary>
      <Router>
        <div className="min-h-screen bg-background">
          <Navigation
            isLoggedIn={isLoggedIn}
            user={user}
            loading={loading}
            onLogin={() => setShowLoginModal(true)}
            onLogout={handleSignOut}
          />
          <AnimatePresence mode="wait">
            <Routes>
              <Route path="/" element={<LandingPage onLogin={() => setShowLoginModal(true)} />} />
              <Route path="/dashboard" element={isLoggedIn ? <Dashboard /> : <LandingPage onLogin={() => setShowLoginModal(true)} />} />
              <Route path="/profile" element={isLoggedIn ? <Profile /> : <LandingPage onLogin={() => setShowLoginModal(true)} />} />
              <Route path="/analysis" element={isLoggedIn ? <Analysis /> : <LandingPage onLogin={() => setShowLoginModal(true)} />} />
              <Route path="/about" element={<About />} />
              <Route path="/features" element={<Features />} />
              <Route path="/faq" element={<FAQ />} />
              <Route path="/onboarding" element={<Onboarding />} />
              <Route path="*" element={<NotFound />} />
            </Routes>
          </AnimatePresence>
          {showLoginModal && (
            <LoginModal
              open={showLoginModal}
              onClose={() => setShowLoginModal(false)}
              onLogin={handleLoginSuccess}
            />
          )}
          <Toaster />
        </div>
      </Router>
    </ErrorBoundary>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <AppContent />
    </AuthProvider>
  );
}