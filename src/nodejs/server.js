// Main Express Server for GPT-OSS Fine-tuning Orchestration
const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const compression = require('compression');
const winston = require('winston');
const path = require('path');
require('dotenv').config();

// Import custom modules
const jobRoutes = require('./routes/jobRoutes');
const modelRoutes = require('./routes/modelRoutes');
const statusRoutes = require('./routes/statusRoutes');
const WebSocketServer = require('./services/websocketService');
const JobManager = require('./services/jobManager');
const QueueManager = require('./services/queueManager');

// Create Express app
const app = express();

// Configure Winston logger
const logger = winston.createLogger({
  level: process.env.LOG_LEVEL || 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.errors({ stack: true }),
    winston.format.json()
  ),
  defaultMeta: { service: 'gpt-oss-orchestrator' },
  transports: [
    new winston.transports.File({ filename: 'logs/error.log', level: 'error' }),
    new winston.transports.File({ filename: 'logs/combined.log' }),
    new winston.transports.Console({
      format: winston.format.simple()
    })
  ],
});

// Global error handler
process.on('uncaughtException', (error) => {
  logger.error('Uncaught Exception:', error);
  process.exit(1);
});

process.on('unhandledRejection', (reason, promise) => {
  logger.error('Unhandled Rejection at:', promise, 'reason:', reason);
});

// Initialize services
const jobManager = new JobManager(logger);
const queueManager = new QueueManager(logger);
const wsServer = new WebSocketServer(logger);

// Middleware
app.use(helmet({
  contentSecurityPolicy: {
    directives: {
      defaultSrc: ["'self'"],
      styleSrc: ["'self'", "'unsafe-inline'"],
      scriptSrc: ["'self'", "'unsafe-inline'"],
      imgSrc: ["'self'", "data:", "https:"],
      connectSrc: ["'self'", "ws:", "wss:"]
    }
  }
}));

app.use(compression());
app.use(cors({
  origin: process.env.CORS_ORIGIN || 'http://localhost:3000',
  credentials: true
}));

app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// Request logging middleware
app.use((req, res, next) => {
  logger.info(`${req.method} ${req.path}`, {
    ip: req.ip,
    userAgent: req.get('User-Agent'),
    timestamp: new Date().toISOString()
  });
  next();
});

// Serve static files (frontend)
app.use(express.static(path.join(__dirname, '../frontend')));

// API Routes
app.use('/api/jobs', jobRoutes(jobManager, queueManager, wsServer));
app.use('/api/models', modelRoutes(jobManager));
app.use('/api/status', statusRoutes(jobManager, queueManager));

// Health check endpoint
app.get('/api/health', (req, res) => {
  res.json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    uptime: process.uptime(),
    memory: process.memoryUsage(),
    version: process.env.npm_package_version || '1.0.0'
  });
});

// API Documentation endpoint
app.get('/api/docs', (req, res) => {
  res.json({
    title: 'GPT-OSS Fine-tuning Orchestrator API',
    version: '1.0.0',
    description: 'RESTful API for orchestrating GPT-OSS 20B fine-tuning with sequential dataset processing',
    endpoints: {
      'POST /api/jobs/finetune': 'Submit a new multi-dataset fine-tuning job',
      'GET /api/jobs/:jobId': 'Get job status and progress',
      'DELETE /api/jobs/:jobId': 'Cancel a running job',
      'GET /api/jobs': 'List all jobs',
      'GET /api/models': 'List available fine-tuned models',
      'POST /api/models/:modelId/deploy': 'Deploy a model to inference endpoint',
      'GET /api/status/queue': 'Get current queue status',
      'GET /api/health': 'Health check endpoint',
      'WebSocket /ws/jobs/:jobId': 'Real-time job progress updates'
    },
    datasets: {
      'Travel Conversations': {
        source: 'soniawmeyer/travel-conversations-finetuning',
        file: 'conversational_sample_processed_with_topic.csv',
        size: '680MB',
        priority: 1
      },
      'Travel QA': {
        source: 'soniawmeyer/travel-conversations-finetuning',
        file: 'travel_QA_processed_with_topic.csv',
        size: '147MB',
        priority: 2
      }
    }
  });
});

// Serve frontend for all other routes
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, '../frontend/index.html'));
});

// Error handling middleware
app.use((err, req, res, next) => {
  logger.error('Express error:', err);
  
  res.status(err.status || 500).json({
    error: {
      message: err.message || 'Internal server error',
      status: err.status || 500,
      timestamp: new Date().toISOString()
    }
  });
});

// 404 handler
app.use((req, res) => {
  res.status(404).json({
    error: {
      message: 'Endpoint not found',
      status: 404,
      path: req.path,
      timestamp: new Date().toISOString()
    }
  });
});

// Start server
const PORT = process.env.PORT || 3000;
const WEBSOCKET_PORT = process.env.WEBSOCKET_PORT || 3001;

app.listen(PORT, () => {
  logger.info(`🚀 GPT-OSS Fine-tuning Orchestrator started on port ${PORT}`);
  logger.info(`📊 Frontend available at: http://localhost:${PORT}`);
  logger.info(`📡 API docs available at: http://localhost:${PORT}/api/docs`);
  
  // Start WebSocket server
  wsServer.start(WEBSOCKET_PORT);
  logger.info(`🔌 WebSocket server started on port ${WEBSOCKET_PORT}`);
});

// Graceful shutdown
process.on('SIGTERM', async () => {
  logger.info('SIGTERM received, shutting down gracefully');
  
  // Stop accepting new requests
  server.close(() => {
    logger.info('Process terminated');
    process.exit(0);
  });
});

process.on('SIGINT', async () => {
  logger.info('SIGINT received, shutting down gracefully');
  
  // Cancel all running jobs
  await jobManager.cancelAllJobs();
  
  // Close WebSocket server
  wsServer.close();
  
  process.exit(0);
});

module.exports = app;