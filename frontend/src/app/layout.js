'use client';

import { Inter, Poppins } from 'next/font/google';
import './globals.css';
import { AuthProvider } from '../contexts/AuthContext';
import { Toaster } from 'react-hot-toast';
import Navbar from '../components/layout/Navbar';

const inter = Inter({ 
  subsets: ['latin'],
  variable: '--font-inter',
  display: 'swap'
});

const poppins = Poppins({ 
  subsets: ['latin'],
  weight: ['300', '400', '500', '600', '700', '800', '900'],
  variable: '--font-poppins',
  display: 'swap'
});

export const metadata = {
  title: 'CareerAI - AI-Powered Career Recommendations',
  description: 'Transform your career with AI-powered job recommendations, skill analysis, and personalized learning paths. The future of career development is here.',
  keywords: 'AI career, job recommendations, skill analysis, career development, machine learning, job search',
  authors: [{ name: 'CareerAI Team' }],
  creator: 'CareerAI',
  publisher: 'CareerAI',
  formatDetection: {
    email: false,
    address: false,
    telephone: false,
  },
  metadataBase: new URL('https://careerai.com'),
  alternates: {
    canonical: '/',
  },
  openGraph: {
    title: 'CareerAI - AI-Powered Career Recommendations',
    description: 'Transform your career with AI-powered job recommendations, skill analysis, and personalized learning paths.',
    url: 'https://careerai.com',
    siteName: 'CareerAI',
    images: [
      {
        url: '/og-image.jpg',
        width: 1200,
        height: 630,
        alt: 'CareerAI - AI-Powered Career Platform',
      },
    ],
    locale: 'en_US',
    type: 'website',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'CareerAI - AI-Powered Career Recommendations',
    description: 'Transform your career with AI-powered job recommendations, skill analysis, and personalized learning paths.',
    images: ['/twitter-image.jpg'],
    creator: '@careerai',
  },
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      'max-video-preview': -1,
      'max-image-preview': 'large',
      'max-snippet': -1,
    },
  },
  verification: {
    google: 'your-google-verification-code',
  },
};

export default function RootLayout({ children }) {
  return (
    <html lang="en" className={`${inter.variable} ${poppins.variable}`}>
      <head>
        <link rel="icon" href="/favicon.ico" />
        <link rel="apple-touch-icon" sizes="180x180" href="/apple-touch-icon.png" />
        <link rel="icon" type="image/png" sizes="32x32" href="/favicon-32x32.png" />
        <link rel="icon" type="image/png" sizes="16x16" href="/favicon-16x16.png" />
        <link rel="manifest" href="/site.webmanifest" />
        <meta name="theme-color" content="#0ea5e9" />
        <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=5" />
      </head>
      <body className={`${inter.className} antialiased bg-dark-900 text-white overflow-x-hidden`}>
        <AuthProvider>
          <div className="relative min-h-screen">
            <Navbar />
            <main className="relative z-10">
              {children}
            </main>
            <Toaster
              position="top-right"
              toastOptions={{
                duration: 4000,
                style: {
                  background: 'rgba(0, 0, 0, 0.8)',
                  color: '#fff',
                  backdropFilter: 'blur(10px)',
                  border: '1px solid rgba(255, 255, 255, 0.1)',
                  borderRadius: '12px',
                },
                success: {
                  iconTheme: {
                    primary: '#10b981',
                    secondary: '#fff',
                  },
                },
                error: {
                  iconTheme: {
                    primary: '#ef4444',
                    secondary: '#fff',
                  },
                },
              }}
            />
          </div>
        </AuthProvider>
      </body>
    </html>
  );
}