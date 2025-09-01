import { Inter, Poppins } from 'next/font/google';
import './globals.css';
import ClientLayout from '../components/layout/ClientLayout';

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
        <meta name="theme-color" content="#0ea5e9" />
        <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=5" />
      </head>
      <body className={`${inter.className} antialiased bg-white dark:bg-gray-900 text-gray-900 dark:text-white overflow-x-hidden transition-colors duration-300`}>
        <ClientLayout>
          {children}
        </ClientLayout>
      </body>
    </html>
  );
}