#!/usr/bin/env node

/**
 * Frontend setup script to ensure all dependencies are installed
 * and the development environment is ready
 */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

console.log('🚀 Setting up AI Career Recommender Frontend...\n');

// Check if package.json exists
if (!fs.existsSync('package.json')) {
    console.error('❌ package.json not found. Please run this script from the frontend directory.');
    process.exit(1);
}

try {
    // Install dependencies
    console.log('📦 Installing dependencies...');
    execSync('npm install', { stdio: 'inherit' });
    
    // Check if .env file exists
    if (!fs.existsSync('.env')) {
        console.log('⚠️  .env file not found. Creating from template...');
        const envTemplate = `# Backend API Configuration
VITE_API_BASE_URL=http://localhost:8000

# Supabase Configuration (for data storage only)
VITE_SUPABASE_URL=https://bmhvwzqadllsyncnyhyw.supabase.co
VITE_SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJtaHZ3enFhZGxsc3luY255aHl3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTc4NDE0ODAsImV4cCI6MjA3MzQxNzQ4MH0.DYSVsBjkiYfLEv1dJfECT2RF8XS2hnSwDL9iIW799co
`;
        fs.writeFileSync('.env', envTemplate);
        console.log('✅ .env file created');
    }
    
    // Verify TypeScript configuration
    if (!fs.existsSync('tsconfig.json')) {
        console.log('⚠️  tsconfig.json not found. This might cause TypeScript issues.');
    } else {
        console.log('✅ TypeScript configuration found');
    }
    
    // Check if all required directories exist
    const requiredDirs = ['src', 'src/components', 'src/services', 'src/contexts'];
    requiredDirs.forEach(dir => {
        if (!fs.existsSync(dir)) {
            console.log(`📁 Creating directory: ${dir}`);
            fs.mkdirSync(dir, { recursive: true });
        }
    });
    
    console.log('\n✅ Frontend setup completed successfully!');
    console.log('\n🎯 Next steps:');
    console.log('1. Make sure the backend is running on http://localhost:8000');
    console.log('2. Run "npm run dev" to start the development server');
    console.log('3. Open http://localhost:3000 in your browser');
    console.log('\n📚 Demo credentials:');
    console.log('   Email: demo@aicareer.com');
    console.log('   Password: secret');
    
} catch (error) {
    console.error('❌ Setup failed:', error.message);
    process.exit(1);
}