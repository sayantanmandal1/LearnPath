# AI Career Recommender Backend

FastAPI-based backend for the AI-Powered Career & Learning Path Recommender platform.

## Features

- **FastAPI Framework**: Modern, fast web framework for building APIs
- **JWT Authentication**: Secure user authentication with access and refresh tokens
- **PostgreSQL Database**: Robust relational database with async support
- **Redis Caching**: High-performance caching and session management
- **Structured Logging**: Comprehensive logging with structlog
- **Prometheus Metrics**: Built-in monitoring and metrics collection
- **Docker Support**: Containerized deployment with Docker Compose

## Quick Start

### Using Docker Compose (Recommended)

1. Clone the repository and navigate to the project root
2. Copy environment file:
   ```bash
   cp backend/.env.example backend/.env
   ```
3. Start all services:
   ```bash
   docker-compose up -d
   ```
4. The API will be available at http://localhost:8000

### Local Development

1. Install Python 3.11+
2. Install dependencies:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```
3. Set up PostgreSQL and Redis locally
4. Copy and configure environment:
   ```bash
   cp .env.example .env
   # Edit .env with your database and Redis URLs
   ```
5. Run database migrations:
   ```bash
   alembic upgrade head
   ```
6. Start the development server:
   ```bash
   uvicorn app.main:app --reload
   ```

## API Documentation

Once the server is running, visit:
- **Interactive API docs**: http://localhost:8000/api/v1/docs
- **ReDoc documentation**: http://localhost:8000/api/v1/redoc
- **OpenAPI JSON**: http://localhost:8000/api/v1/openapi.json

## Project Structure

```
backend/
├── app/
│   ├── api/                 # API endpoints
│   │   └── v1/
│   │       ├── endpoints/   # Route handlers
│   │       └── router.py    # Main router
│   ├── core/                # Core functionality
│   │   ├── config.py        # Configuration settings
│   │   ├── database.py      # Database setup
│   │   ├── redis.py         # Redis setup
│   │   ├── security.py      # Authentication utilities
│   │   └── logging.py       # Logging configuration
│   ├── middleware/          # Custom middleware
│   ├── models/              # SQLAlchemy models
│   ├── schemas/             # Pydantic schemas
│   ├── services/            # Business logic
│   └── main.py              # FastAPI application
├── alembic/                 # Database migrations
├── scripts/                 # Utility scripts
├── requirements.txt         # Python dependencies
└── Dockerfile              # Docker configuration
```

## Environment Variables

Key environment variables (see `.env.example` for full list):

- `DATABASE_URL`: PostgreSQL connection string
- `REDIS_URL`: Redis connection string
- `SECRET_KEY`: Application secret key
- `JWT_SECRET_KEY`: JWT signing key
- `DEBUG`: Enable debug mode

## Database Migrations

Create a new migration:
```bash
alembic revision --autogenerate -m "Description"
```

Apply migrations:
```bash
alembic upgrade head
```

## Testing

Run tests:
```bash
pytest
```

Run with coverage:
```bash
pytest --cov=app
```

## Monitoring

- **Health check**: GET `/health`
- **Metrics**: GET `/metrics` (Prometheus format)
- **Logs**: Structured JSON logs to stdout

## Security

- JWT-based authentication with access and refresh tokens
- Password hashing with bcrypt
- Input validation with Pydantic
- SQL injection protection with SQLAlchemy
- CORS configuration
- Rate limiting (planned)

## Contributing

1. Follow PEP 8 style guidelines
2. Add tests for new features
3. Update documentation
4. Use conventional commit messages