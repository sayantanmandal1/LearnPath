# AI Career Recommender - Complete Learning Path Platform

A comprehensive AI-powered platform that analyzes student profiles and provides personalized career recommendations, learning paths, and job matching. Built with FastAPI, React, and PostgreSQL.

## 🚀 Features

### Core Functionality
- **Personalized Career Analysis**: AI-driven career path recommendations based on skills, interests, and market trends
- **Learning Path Generation**: Customized learning roadmaps with resources from multiple platforms
- **Job Matching**: Real-time job recommendations with compatibility scoring
- **Skill Gap Analysis**: Identify missing skills and get targeted improvement suggestions
- **Market Insights**: Live job market trends and salary predictions
- **Resume Analysis**: AI-powered resume parsing and optimization suggestions

### Advanced Features
- **Multi-Platform Integration**: GitHub, LeetCode, LinkedIn profile analysis
- **Real-time Job Scraping**: Live job data from major platforms
- **Interactive Visualizations**: Skill radar charts, career trajectory roadmaps
- **PDF Report Generation**: Comprehensive career analysis reports
- **Performance Monitoring**: System health and analytics tracking
- **Vector Search**: Semantic job and skill matching

## 🏗️ Architecture

### Backend (FastAPI)
- **API Layer**: RESTful APIs with comprehensive documentation
- **ML Engine**: Machine learning models for recommendations and analysis
- **Data Pipeline**: Automated data collection and processing
- **Vector Database**: Semantic search capabilities
- **Background Tasks**: Celery-based task processing
- **Monitoring**: Prometheus metrics and alerting

### Frontend (React + TypeScript)
- **Modern UI**: Responsive design with Tailwind CSS
- **Interactive Components**: Dynamic charts and visualizations
- **Real-time Updates**: WebSocket connections for live data
- **Authentication**: JWT-based secure authentication
- **State Management**: Context API and custom hooks

### Database
- **PostgreSQL**: Primary data storage with advanced indexing
- **Redis**: Caching and session management
- **Vector Storage**: Embeddings for semantic search

## 🛠️ Quick Start

### Prerequisites
- Docker and Docker Compose
- Node.js 18+ (for frontend development)
- Python 3.11+ (for backend development)

### 1. Clone and Setup
```bash
git clone <repository-url>
cd LearnPath
```

### 2. Setup Environment Variables
```bash
# Create secure .env files from templates
python setup_env.py

# Or manually copy and edit templates
cp backend/.env.template backend/.env
cp frontend/.env.template frontend/.env
# Then edit the .env files with your actual API keys
```

### 3. Start Backend Services
```bash
# Start all services (PostgreSQL, Redis, Backend API)
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f backend
```

### 3. Initialize Database
The system automatically initializes with:
- **123 Skills** across 10 categories (Programming, Web Dev, AI/ML, etc.)
- **10 Sample Jobs** from various companies and roles
- **Demo User**: `demo@aicareer.com` / `secret`

### 4. Setup Frontend
```bash
cd frontend
node setup.js
npm run dev
```

### 5. Access the Application
- **Frontend**: http://localhost:3000
- **API Documentation**: http://localhost:8000/api/v1/docs
- **Health Check**: http://localhost:8000/health

## 📊 Demo Data

### Sample Skills Categories
- **Programming Languages**: Python, JavaScript, Java, C++, Go, Rust
- **Web Development**: React, Angular, Node.js, Django, FastAPI
- **Data Science**: Machine Learning, TensorFlow, PyTorch, Pandas
- **Cloud & DevOps**: AWS, Docker, Kubernetes, Jenkins
- **Cybersecurity**: Network Security, Penetration Testing, CISSP

### Sample Job Postings
- Senior Full Stack Developer @ TechCorp Solutions ($120k-$180k)
- Data Scientist @ DataInsights Inc ($100k-$150k)
- Machine Learning Engineer @ AI Innovations Lab ($130k-$200k)
- DevOps Engineer @ CloudFirst Technologies ($110k-$160k)
- And 6 more diverse roles...

## 🔧 Development

### Backend Development
```bash
# Enter backend container
docker-compose exec backend bash

# Run migrations
alembic upgrade head

# Run tests
pytest

# Add new migration
alembic revision --autogenerate -m "description"
```

### Frontend Development
```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Run tests
npm test
```

### Database Management
```bash
# Connect to PostgreSQL
docker-compose exec postgres psql -U postgres -d ai_career_db

# View tables
\dt

# Check data
SELECT COUNT(*) FROM skills;
SELECT COUNT(*) FROM job_postings;
SELECT * FROM users WHERE email = 'demo@aicareer.com';
```

## 📈 API Endpoints

### Authentication
- `POST /api/v1/auth/register` - User registration
- `POST /api/v1/auth/login` - User login
- `POST /api/v1/auth/refresh` - Token refresh

### Core Features
- `GET /api/v1/profiles/` - User profiles
- `GET /api/v1/recommendations/` - Career recommendations
- `GET /api/v1/learning-paths/` - Learning path generation
- `GET /api/v1/job-market/` - Job market analysis
- `GET /api/v1/analytics/` - User analytics

### Advanced Features
- `POST /api/v1/vector-search/` - Semantic search
- `GET /api/v1/market-insights/` - Market trends
- `POST /api/v1/resume/upload` - Resume analysis
- `GET /api/v1/career-trajectory/` - Career planning

## 🔒 Security Features

- JWT-based authentication with refresh tokens
- Password hashing with bcrypt
- Rate limiting and request validation
- CORS configuration for cross-origin requests
- Input sanitization and SQL injection prevention
- Audit logging for sensitive operations

## 📊 Monitoring & Analytics

- **Health Checks**: Automated service health monitoring
- **Performance Metrics**: Response time and throughput tracking
- **Error Tracking**: Comprehensive error logging and alerting
- **Usage Analytics**: User behavior and feature usage tracking
- **System Alerts**: Automated alerts for system issues

## 🚀 Production Deployment

### Environment Variables
```bash
# Use the setup script to create secure .env files
python setup_env.py

# Or manually create from templates:
# backend/.env - Contains database, API keys, JWT secrets
# frontend/.env - Contains API URLs and Supabase config

# Key variables to update:
# GEMINI_API_KEY=your-actual-gemini-api-key
# VITE_SUPABASE_URL=your-supabase-project-url
# VITE_SUPABASE_ANON_KEY=your-supabase-anon-key
```

### Docker Production
```bash
# Build production images
docker-compose -f docker-compose.prod.yml build

# Deploy
docker-compose -f docker-compose.prod.yml up -d
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: Check `/api/v1/docs` for API documentation
- **Issues**: Report bugs and feature requests via GitHub Issues
- **Demo**: Use `demo@aicareer.com` / `secret` for testing

## 🎯 Roadmap

- [ ] Mobile app development
- [ ] Advanced ML model training
- [ ] Integration with more job platforms
- [ ] Real-time collaboration features
- [ ] Advanced analytics dashboard
- [ ] Multi-language support

---

**Built with ❤️ for students and career professionals**