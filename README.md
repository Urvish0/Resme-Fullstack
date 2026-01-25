# Resme – ATS Resume & Cover Letter Optimizer

A **full-stack, production-oriented AI application** that optimizes resumes for ATS systems, generates tailored cover letters, and extracts job descriptions from any website using advanced web scraping and LLM workflows.

## 🎯 What It Does

- **Resume Optimization**: Analyzes your resume against job descriptions and optimizes it for ATS (Applicant Tracking System) compatibility
- **Cover Letter Generation**: Generates tailored, compelling cover letters based on your resume and the job description
- **Smart Job Description Extraction**: Extracts job descriptions from URLs, including JavaScript-heavy sites (LinkedIn, Qualcomm, etc.)
- **ATS Scoring**: Provides before/after ATS compatibility scores
- **Async Job Processing**: Handles long-running operations with progress tracking and polling

## 🏗️ Architecture

### Monorepo Structure

```
Resme_Fullstack/
├── backend/              # FastAPI + LangGraph + Redis
├── frontend/             # Next.js 16 (App Router)
├── docker-compose.yaml   # Multi-container orchestration
├── requirements.txt      # Python dependencies
├── PROJECT_CONTEXT.md    # Detailed project documentation
└── outputs/              # Generated cover letters
```

## 🛠️ Tech Stack

### Backend
- **FastAPI** (Python 3.13)
- **LangGraph** for agentic AI workflows
- **Groq LLM** (llama-3.1-8b-instant)
- **Playwright** for universal web scraping
- **Redis** for caching and job state management
- **Prometheus** for observability

### Frontend
- **Next.js 16** (App Router)
- **React 19**
- **TypeScript**
- **Tailwind CSS**
- **ReactMarkdown** for rendered markdown display

## 🚀 Quick Start

### Prerequisites
- Python 3.13+
- Node.js 18+
- Docker & Docker Compose
- Groq API key (free at [groq.com](https://groq.com))

### Environment Variables

Create a `.env` file in the `backend/` directory:

```
GROQ_API_KEY=your_groq_api_key_here
REDIS_URL=redis://localhost:6379
```

Frontend env vars in `frontend/.env.local`:

```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### Run Locally

#### Option 1: Using Docker Compose (Recommended)

```bash
docker compose up --build
```

- Backend API: http://localhost:8000
- Frontend: http://localhost:3000
- API Docs: http://localhost:8000/docs

#### Option 2: Manual Setup

**Backend:**
```bash
cd backend
pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

## 💡 Key Features

### Resume Optimization Workflow
1. Upload your resume (PDF, DOC, DOCX, or paste text)
2. Provide job description (paste text or enter URL)
3. System automatically:
   - Extracts keywords and requirements
   - Analyzes ATS compatibility
   - Generates optimized resume with improved keywords
   - Creates tailored cover letter

### Universal Job Description Scraping
- Handles JavaScript-heavy sites using Playwright
- Multi-phase extraction: JSON-LD → Semantic HTML → Fallback text
- Works on LinkedIn, Qualcomm, Indeed, and any other job board

### Async Job Processing
- Long-running workflows processed in background
- Progress tracking via polling endpoint
- Idempotency keys for safe retries
- Redis-backed job state

## 📊 API Endpoints

### Resume Optimization
- `POST /optimize` - Synchronous optimization
- `POST /optimize/async` - Asynchronous job submission
- `GET /jobs/{job_id}` - Check job status
- `GET /jobs/{job_id}/result` - Retrieve results

### Job Description Extraction
- `POST /extract-jd` - Extract JD from URL

See [http://localhost:8000/docs](http://localhost:8000/docs) for full API documentation.

## 🎨 Frontend Features

- **Resume Upload**: Support for PDF, DOCX, and DOC files
- **Real-time Progress**: Visual progress bar for async jobs
- **Before/After Scores**: Display ATS scores improvement
- **Tabbed Interface**: View optimized resume and cover letter in separate tabs
- **Download & Copy**: Export results as markdown
- **Responsive Design**: Works on desktop and mobile

## 🔧 Project Status

### ✅ Completed
- Core resume optimization pipeline
- Async job orchestration
- Universal JD scraping (Playwright-based)
- ATS scoring pipeline
- Markdown rendering
- Dockerized backend
- Redis integration
- FastAPI production setup

### 🟡 In Progress
- Frontend Dockerization
- GitHub Actions CI pipeline

### 🔵 Roadmap
- **Auth**: User authentication (Clerk/Auth0)
- **SaaS Features**: Credits, billing, rate limiting
- **Advanced AI**: Resume critique agent, skill gap analysis
- **Export**: PDF/DOCX generation
- **Portfolio Features**: Resume versioning, job history

## 📚 Documentation

- [PROJECT_CONTEXT.md](./PROJECT_CONTEXT.md) - Comprehensive project documentation
- [BACKEND_STRUCTURE.md](./BACKEND_STRUCTURE.md) - Backend architecture details
- [FOLDER_STRUCTURE.md](./FOLDER_STRUCTURE.md) - Directory structure
- API Docs: http://localhost:8000/docs (when running)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 💬 Support

For issues, questions, or suggestions, please open an issue on GitHub or refer to [PROJECT_CONTEXT.md](./PROJECT_CONTEXT.md) for detailed documentation.

---

**Built with ❤️ as a production-grade AI portfolio project**
