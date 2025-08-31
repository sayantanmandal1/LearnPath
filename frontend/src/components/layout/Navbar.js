'use client';

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useAuth } from '../../contexts/AuthContext';
import AnimatedButton from '../ui/AnimatedButton';
import GlassCard from '../ui/GlassCard';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import {
  HomeIcon,
  UserIcon,
  BriefcaseIcon,
  ChartBarIcon,
  AcademicCapIcon,
  CogIcon,
  ArrowRightOnRectangleIcon as LogoutIcon,
  Bars3Icon as MenuIcon,
  XMarkIcon as XIcon,
  BellIcon,
  MagnifyingGlassIcon as SearchIcon
} from '@heroicons/react/24/outline';
import anime from 'animejs';

const Navbar = () => {
  const { user, userProfile, logout } = useAuth();
  const router = useRouter();
  const [isOpen, setIsOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  const [notifications, setNotifications] = useState(3);

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 20);
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  useEffect(() => {
    // Animate navbar on mount
    anime({
      targets: '.navbar-item',
      translateY: [-20, 0],
      opacity: [0, 1],
      duration: 800,
      delay: anime.stagger(100),
      easing: 'easeOutExpo'
    });
  }, []);

  const navItems = [
    { name: 'Dashboard', href: '/dashboard', icon: HomeIcon },
    { name: 'Profile', href: '/profile', icon: UserIcon },
    { name: 'Jobs', href: '/jobs', icon: BriefcaseIcon },
    { name: 'Analytics', href: '/analytics', icon: ChartBarIcon },
    { name: 'Learning', href: '/learning', icon: AcademicCapIcon },
  ];

  const handleLogout = async () => {
    await logout();
    router.push('/');
  };

  const NavLink = ({ item, mobile = false }) => (
    <Link href={item.href}>
      <motion.div
        className={`
          flex items-center space-x-2 px-4 py-2 rounded-lg
          text-gray-300 hover:text-white hover:bg-white/10
          transition-all duration-300 cursor-pointer
          ${mobile ? 'w-full justify-start' : ''}
        `}
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
      >
        <item.icon className="w-5 h-5" />
        <span className={mobile ? 'block' : 'hidden lg:block'}>{item.name}</span>
      </motion.div>
    </Link>
  );

  const UserMenu = () => (
    <motion.div
      className="relative"
      initial={{ opacity: 0, scale: 0.8 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ type: "spring", stiffness: 300, damping: 30 }}
    >
      <motion.div
        className="flex items-center space-x-3 cursor-pointer"
        whileHover={{ scale: 1.05 }}
        onClick={() => setIsOpen(!isOpen)}
      >
        <div className="relative">
          <motion.div
            className="w-10 h-10 rounded-full bg-gradient-to-r from-primary-400 to-secondary-400 flex items-center justify-center"
            whileHover={{ rotate: 360 }}
            transition={{ duration: 0.5 }}
          >
            {userProfile?.photoURL ? (
              <img
                src={userProfile.photoURL}
                alt="Profile"
                className="w-full h-full rounded-full object-cover"
              />
            ) : (
              <UserIcon className="w-6 h-6 text-white" />
            )}
          </motion.div>
          {notifications > 0 && (
            <motion.div
              className="absolute -top-1 -right-1 w-5 h-5 bg-red-500 rounded-full flex items-center justify-center text-xs text-white"
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ type: "spring", stiffness: 500, damping: 25 }}
            >
              {notifications}
            </motion.div>
          )}
        </div>
        <div className="hidden lg:block">
          <p className="text-white font-medium">
            {userProfile?.profile?.firstName || user?.displayName || 'User'}
          </p>
          <p className="text-gray-400 text-sm">
            {userProfile?.profile?.title || 'Professional'}
          </p>
        </div>
      </motion.div>

      <AnimatePresence>
        {isOpen && (
          <motion.div
            className="absolute right-0 mt-2 w-64 z-50"
            initial={{ opacity: 0, y: -10, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -10, scale: 0.95 }}
            transition={{ type: "spring", stiffness: 300, damping: 30 }}
          >
            <GlassCard className="p-4">
              <div className="flex items-center space-x-3 mb-4 pb-4 border-b border-white/10">
                <div className="w-12 h-12 rounded-full bg-gradient-to-r from-primary-400 to-secondary-400 flex items-center justify-center">
                  {userProfile?.photoURL ? (
                    <img
                      src={userProfile.photoURL}
                      alt="Profile"
                      className="w-full h-full rounded-full object-cover"
                    />
                  ) : (
                    <UserIcon className="w-7 h-7 text-white" />
                  )}
                </div>
                <div>
                  <p className="text-white font-medium">
                    {userProfile?.profile?.firstName || user?.displayName || 'User'}
                  </p>
                  <p className="text-gray-400 text-sm">{user?.email}</p>
                </div>
              </div>

              <div className="space-y-2">
                <Link href="/profile">
                  <motion.div
                    className="flex items-center space-x-3 px-3 py-2 rounded-lg hover:bg-white/10 cursor-pointer"
                    whileHover={{ x: 5 }}
                  >
                    <UserIcon className="w-5 h-5 text-gray-400" />
                    <span className="text-gray-300">View Profile</span>
                  </motion.div>
                </Link>

                <Link href="/settings">
                  <motion.div
                    className="flex items-center space-x-3 px-3 py-2 rounded-lg hover:bg-white/10 cursor-pointer"
                    whileHover={{ x: 5 }}
                  >
                    <CogIcon className="w-5 h-5 text-gray-400" />
                    <span className="text-gray-300">Settings</span>
                  </motion.div>
                </Link>

                <Link href="/notifications">
                  <motion.div
                    className="flex items-center space-x-3 px-3 py-2 rounded-lg hover:bg-white/10 cursor-pointer"
                    whileHover={{ x: 5 }}
                  >
                    <BellIcon className="w-5 h-5 text-gray-400" />
                    <span className="text-gray-300">Notifications</span>
                    {notifications > 0 && (
                      <span className="ml-auto bg-red-500 text-white text-xs px-2 py-1 rounded-full">
                        {notifications}
                      </span>
                    )}
                  </motion.div>
                </Link>

                <motion.div
                  className="flex items-center space-x-3 px-3 py-2 rounded-lg hover:bg-red-500/20 cursor-pointer border-t border-white/10 mt-2 pt-2"
                  whileHover={{ x: 5 }}
                  onClick={handleLogout}
                >
                  <LogoutIcon className="w-5 h-5 text-red-400" />
                  <span className="text-red-400">Sign Out</span>
                </motion.div>
              </div>
            </GlassCard>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );

  return (
    <>
      <motion.nav
        className={`
          fixed top-0 left-0 right-0 z-50 transition-all duration-300
          ${scrolled 
            ? 'backdrop-blur-xl bg-black/20 border-b border-white/10' 
            : 'bg-transparent'
          }
        `}
        initial={{ y: -100 }}
        animate={{ y: 0 }}
        transition={{ type: "spring", stiffness: 300, damping: 30 }}
      >
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            {/* Logo */}
            <Link href="/">
              <motion.div
                className="flex items-center space-x-2 cursor-pointer navbar-item"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <div className="w-10 h-10 bg-gradient-to-r from-primary-500 to-secondary-500 rounded-xl flex items-center justify-center">
                  <span className="text-white font-bold text-xl">AI</span>
                </div>
                <span className="text-white font-bold text-xl hidden sm:block">
                  CareerAI
                </span>
              </motion.div>
            </Link>

            {/* Desktop Navigation */}
            {user && (
              <div className="hidden md:flex items-center space-x-1">
                {navItems.map((item, index) => (
                  <div key={item.name} className="navbar-item">
                    <NavLink item={item} />
                  </div>
                ))}
              </div>
            )}

            {/* Search Bar */}
            {user && (
              <div className="hidden lg:flex items-center navbar-item">
                <div className="relative">
                  <SearchIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
                  <input
                    type="text"
                    placeholder="Search jobs, skills..."
                    className="pl-10 pr-4 py-2 bg-white/10 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent backdrop-blur-sm"
                  />
                </div>
              </div>
            )}

            {/* User Menu or Auth Buttons */}
            <div className="flex items-center space-x-4">
              {user ? (
                <UserMenu />
              ) : (
                <div className="flex items-center space-x-3 navbar-item">
                  <Link href="/auth/signin">
                    <AnimatedButton variant="ghost" size="sm">
                      Sign In
                    </AnimatedButton>
                  </Link>
                  <Link href="/auth/signup">
                    <AnimatedButton variant="primary" size="sm" glow>
                      Get Started
                    </AnimatedButton>
                  </Link>
                </div>
              )}

              {/* Mobile Menu Button */}
              {user && (
                <motion.button
                  className="md:hidden text-white p-2"
                  whileTap={{ scale: 0.95 }}
                  onClick={() => setIsOpen(!isOpen)}
                >
                  {isOpen ? (
                    <XIcon className="w-6 h-6" />
                  ) : (
                    <MenuIcon className="w-6 h-6" />
                  )}
                </motion.button>
              )}
            </div>
          </div>
        </div>

        {/* Mobile Menu */}
        <AnimatePresence>
          {isOpen && user && (
            <motion.div
              className="md:hidden"
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.3 }}
            >
              <GlassCard className="mx-4 mb-4 p-4">
                <div className="space-y-2">
                  {navItems.map((item) => (
                    <NavLink key={item.name} item={item} mobile />
                  ))}
                  
                  <div className="border-t border-white/10 pt-2 mt-2">
                    <Link href="/settings">
                      <motion.div
                        className="flex items-center space-x-2 px-4 py-2 rounded-lg text-gray-300 hover:text-white hover:bg-white/10 cursor-pointer"
                        whileHover={{ x: 5 }}
                      >
                        <CogIcon className="w-5 h-5" />
                        <span>Settings</span>
                      </motion.div>
                    </Link>
                    
                    <motion.div
                      className="flex items-center space-x-2 px-4 py-2 rounded-lg text-red-400 hover:bg-red-500/20 cursor-pointer"
                      whileHover={{ x: 5 }}
                      onClick={handleLogout}
                    >
                      <LogoutIcon className="w-5 h-5" />
                      <span>Sign Out</span>
                    </motion.div>
                  </div>
                </div>
              </GlassCard>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.nav>

      {/* Spacer */}
      <div className="h-16" />
    </>
  );
};

export default Navbar;