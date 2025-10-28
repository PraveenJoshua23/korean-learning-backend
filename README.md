# Korean Learning Platform API

A comprehensive FastAPI backend for a Korean language learning platform with AI chat integration, progress tracking, and payment verification system.

## Features

- **User Authentication**: JWT-based authentication with secure password hashing
- **Learning Content**: Vocabulary and grammar management with difficulty levels
- **Progress Tracking**: User progress monitoring with streak calculation
- **AI Chat**: Integration with Anthropic Claude for Korean language tutoring
- **PDF Processing**: Extract vocabulary from PDF files and import to database
- **Payment System**: UPI payment verification for premium features
- **Admin Functions**: Payment approval and user management

## Tech Stack

- **Framework**: FastAPI
- **Database**: SQLAlchemy with SQLite (production-ready for PostgreSQL)
- **Authentication**: JWT with python-jose and bcrypt
- **AI Integration**: Anthropic Claude API
- **PDF Processing**: pdfplumber and PyPDF2
- **Deployment**: Railway-ready configuration

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd KoreanLearningApp

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Setup

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your configuration
nano .env
```

**Required environment variables:**
- `JWT_SECRET`: Secure random string for JWT token signing
- `ANTHROPIC_API_KEY`: Your Anthropic Claude API key

### 3. Database Setup

```bash
# Import sample data (vocabulary and grammar)
python import_data.py --all

# Or import sample data only
python import_data.py --sample

# Import vocabulary from CSV (optional)
python import_data.py --csv your_vocabulary.csv
```

### 4. Run the Application

```bash
# Development server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Production server
uvicorn main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

## API Documentation

Once running, access the interactive API documentation at:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## API Endpoints

### Authentication
- `POST /api/auth/register` - User registration
- `POST /api/auth/login` - User login

### User Management
- `GET /api/user/profile` - Get user profile

### Learning Content
- `GET /api/vocabulary` - List vocabulary (paginated)
- `GET /api/vocabulary/{id}` - Get specific vocabulary item
- `GET /api/grammar` - List grammar points (with category filter)
- `GET /api/grammar/{id}` - Get specific grammar point

### Progress Tracking
- `POST /api/progress` - Update learning progress
- `GET /api/progress` - Get user's progress
- `POST /api/study-session` - Log study time and update streaks

### AI Chat
- `POST /api/chat` - Send message to Korean language tutor
- `GET /api/chat/history` - Get conversation history

### Payment System
- `POST /api/payment/submit` - Submit UPI payment for verification
- `GET /api/payment/status` - Check payment status
- `POST /api/admin/verify-payment/{payment_id}` - Admin: approve payments

## PDF Processing

Extract vocabulary from PDF files using the included script:

```bash
# Extract vocabulary from PDF to CSV
python pdf_to_csv.py input.pdf output.csv

# Use text fallback if table detection fails
python pdf_to_csv.py input.pdf output.csv --text-fallback

# Import the extracted CSV to database
python import_data.py --csv output.csv
```

## Database Models

### User
- Authentication and profile information
- Premium status and streak tracking
- Study time logging

### Vocabulary
- Korean-English word pairs
- Categories and difficulty levels

### Grammar
- Grammar explanations with examples
- Categorized by type and difficulty

### Progress
- User learning progress tracking
- Mastery levels and review counts

### ChatMessage
- AI conversation history
- User and assistant messages

### PaymentVerification
- UPI payment tracking
- Admin verification workflow

## Authentication

All endpoints except `/api/auth/*` require JWT authentication:

```bash
# Login to get token
curl -X POST "http://localhost:8000/api/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "password"}'

# Use token in subsequent requests
curl -X GET "http://localhost:8000/api/user/profile" \
  -H "Authorization: Bearer <your-jwt-token>"
```

## Deployment

### Railway Deployment

1. **Connect to Railway:**
   ```bash
   # Install Railway CLI
   npm install -g @railway/cli
   
   # Login and deploy
   railway login
   railway init
   railway up
   ```

2. **Environment Variables:**
   Set these in Railway dashboard:
   - `JWT_SECRET`
   - `ANTHROPIC_API_KEY`
   - `DATABASE_URL` (for PostgreSQL)

3. **Database Migration:**
   Railway will automatically run the application and create SQLite database. For PostgreSQL:
   ```bash
   # Add PostgreSQL service in Railway dashboard
   # Update DATABASE_URL in environment variables
   # The app will automatically create tables on startup
   ```

### Manual Deployment

1. **Production Environment:**
   ```bash
   # Set environment variables
   export JWT_SECRET="your-production-secret"
   export ANTHROPIC_API_KEY="your-api-key"
   export DATABASE_URL="your-production-database-url"
   
   # Run with production server
   uvicorn main:app --host 0.0.0.0 --port $PORT
   ```

## Development

### Project Structure
```
KoreanLearningApp/
├── main.py              # FastAPI application with all endpoints
├── requirements.txt     # Python dependencies
├── pdf_to_csv.py       # PDF vocabulary extraction script
├── import_data.py      # Database import script
├── .env.example        # Environment variables template
├── .gitignore          # Git ignore file
├── README.md           # This file
├── railway.toml        # Railway deployment config
└── Procfile           # Process file for deployment
```

### Adding New Features

1. **New Endpoints**: Add to `main.py` with proper authentication
2. **Database Models**: Extend SQLAlchemy models in `main.py`
3. **Data Import**: Extend `import_data.py` for new data types
4. **PDF Processing**: Modify `pdf_to_csv.py` for different PDF formats

## Troubleshooting

### Common Issues

1. **Database Connection Error:**
   ```bash
   # Check DATABASE_URL in .env
   # Ensure SQLite file permissions
   # For PostgreSQL, verify connection string
   ```

2. **JWT Token Invalid:**
   ```bash
   # Check JWT_SECRET is set correctly
   # Ensure token is not expired (30-day default)
   ```

3. **Anthropic API Error:**
   ```bash
   # Verify ANTHROPIC_API_KEY is correct
   # Check API quota and billing
   ```

4. **PDF Processing Fails:**
   ```bash
   # Try --text-fallback option
   # Check PDF format and table structure
   # Ensure pdfplumber dependencies are installed
   ```

### Logs

Application logs important events:
- User authentication
- Payment submissions
- Errors and exceptions

Check logs for debugging:
```bash
# Development
python main.py  # Logs to console

# Production
# Check Railway logs or your deployment platform
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Security

- JWT tokens expire after 30 days
- Passwords are hashed with bcrypt
- CORS is configured for security
- Input validation on all endpoints
- No secrets logged or exposed

## License

This project is licensed under the MIT License.

## Support

For issues and questions:
1. Check this README
2. Review API documentation at `/docs`
3. Create an issue in the repository
4. Check logs for error details