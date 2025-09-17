import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { motion } from 'motion/react';
import { Button } from './ui/button';
import { Avatar, AvatarImage, AvatarFallback } from './ui/avatar';
import { 
  DropdownMenu, 
  DropdownMenuContent, 
  DropdownMenuItem, 
  DropdownMenuTrigger 
} from './ui/dropdown-menu';
import { 
  NavigationIcon, 
  User, 
  LogOut, 
  BarChart3, 
  Brain, 
  Settings 
} from 'lucide-react';
import type { User as SupabaseUser } from '@supabase/supabase-js';
import { authUtils } from '../utils/auth';

interface NavigationProps {
  isLoggedIn: boolean;
  user: SupabaseUser | null;
  loading: boolean;
  onLogin: () => void;
  onLogout: () => void;
}

export function Navigation({ isLoggedIn, user, loading, onLogin, onLogout }: NavigationProps) {
  const location = useLocation();

  const navItems = [
    { path: '/', label: 'Home' },
    { path: '/about', label: 'About' },
    { path: '/features', label: 'Features' },
    { path: '/faq', label: 'FAQ' },
  ];

  const dashboardItems = [
    { path: '/dashboard', label: 'Dashboard', icon: BarChart3 },
    { path: '/analysis', label: 'Analysis', icon: Brain },
    { path: '/profile', label: 'Profile', icon: User },
  ];

  return (
    <motion.nav 
      initial={{ y: -100 }}
      animate={{ y: 0 }}
      className="fixed top-0 left-0 right-0 z-50 bg-background/80 backdrop-blur-md border-b border-border"
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Logo */}
          <Link to="/" className="flex items-center space-x-2">
            <motion.div
              whileHover={{ rotate: 360 }}
              transition={{ duration: 0.5 }}
              className="w-8 h-8 bg-primary rounded-lg flex items-center justify-center"
            >
              <NavigationIcon className="w-5 h-5 text-primary-foreground" />
            </motion.div>
            <span className="text-xl font-semibold">CareerPilot</span>
          </Link>

          {/* Navigation Items */}
          <div className="hidden md:flex items-center space-x-8">
            {navItems.map((item) => (
              <Link
                key={item.path}
                to={item.path}
                className={`relative px-3 py-2 transition-colors hover:text-primary ${
                  location.pathname === item.path ? 'text-primary' : 'text-muted-foreground'
                }`}
              >
                {item.label}
                {location.pathname === item.path && (
                  <motion.div
                    layoutId="navbar-indicator"
                    className="absolute bottom-0 left-0 right-0 h-0.5 bg-primary"
                  />
                )}
              </Link>
            ))}
          </div>

          {/* User Section */}
          <div className="flex items-center space-x-4">
            {isLoggedIn ? (
              <>
                {/* Dashboard Navigation */}
                <div className="hidden lg:flex items-center space-x-4">
                  {dashboardItems.map((item) => {
                    const Icon = item.icon;
                    return (
                      <Link
                        key={item.path}
                        to={item.path}
                        className={`flex items-center space-x-2 px-3 py-2 rounded-lg transition-colors hover:bg-accent ${
                          location.pathname === item.path ? 'bg-accent text-accent-foreground' : 'text-muted-foreground'
                        }`}
                      >
                        <Icon className="w-4 h-4" />
                        <span>{item.label}</span>
                      </Link>
                    );
                  })}
                </div>

                {/* User Avatar */}
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button variant="ghost" className="relative h-8 w-8 rounded-full">
                      <Avatar className="h-8 w-8">
                        <AvatarImage src={user?.user_metadata?.avatar_url} alt="User" />
                        <AvatarFallback>
                          {user?.email?.charAt(0).toUpperCase() || 'U'}
                        </AvatarFallback>
                      </Avatar>
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent className="w-56" align="end" forceMount>
                    <div className="px-2 py-1.5 text-sm text-muted-foreground">
                      {user?.email}
                    </div>
                    <DropdownMenuItem asChild>
                      <Link to="/profile" className="flex items-center">
                        <User className="mr-2 h-4 w-4" />
                        Profile
                      </Link>
                    </DropdownMenuItem>
                    <DropdownMenuItem>
                      <Settings className="mr-2 h-4 w-4" />
                      Settings
                    </DropdownMenuItem>
                    <DropdownMenuItem onClick={async () => {
                      const result = await authUtils.signOut();
                      if (result.success) {
                        onLogout();
                      }
                    }}>
                      <LogOut className="mr-2 h-4 w-4" />
                      Log out
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              </>
            ) : (
              <Button 
                onClick={onLogin} 
                disabled={loading}
                className="bg-primary text-primary-foreground hover:bg-primary/90"
              >
                {loading ? 'Loading...' : 'Sign In'}
              </Button>
            )}
          </div>
        </div>
      </div>
    </motion.nav>
  );
}