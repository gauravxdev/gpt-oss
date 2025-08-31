// Job Routes - API endpoints for fine-tuning job management
const express = require('express');
const { body, param, validationResult } = require('express-validator');
const { v4: uuidv4 } = require('uuid');

module.exports = (jobManager, queueManager, wsServer) => {
  const router = express.Router();

  // Validation middleware
  const validateJobSubmission = [
    body('model_name').optional().isString().withMessage('Model name must be a string'),
    body('datasets_config').optional().isArray().withMessage('Datasets config must be an array'),
    body('training_config.max_steps_per_dataset').optional().isInt({ min: 10, max: 10000 }).withMessage('Max steps must be between 10 and 10000'),
    body('training_config.learning_rate').optional().isFloat({ min: 0.00001, max: 0.01 }).withMessage('Learning rate must be between 0.00001 and 0.01'),
    body('training_config.batch_size').optional().isInt({ min: 1, max: 32 }).withMessage('Batch size must be between 1 and 32'),
    body('lora_config.rank').optional().isInt({ min: 4, max: 128 }).withMessage('LoRA rank must be between 4 and 128'),
    body('lora_config.alpha').optional().isInt({ min: 8, max: 256 }).withMessage('LoRA alpha must be between 8 and 256'),
    body('gpu_config.type').optional().isIn(['H100', 'L40S']).withMessage('GPU type must be H100 or L40S'),
    body('gpu_config.timeout_hours').optional().isInt({ min: 1, max: 24 }).withMessage('Timeout must be between 1 and 24 hours'),
  ];

  const validateJobId = [
    param('jobId').isUUID().withMessage('Job ID must be a valid UUID')
  ];

  // POST /api/jobs/finetune - Submit new fine-tuning job
  router.post('/finetune', validateJobSubmission, async (req, res) => {
    try {
      // Check validation errors
      const errors = validationResult(req);
      if (!errors.isEmpty()) {
        return res.status(400).json({
          error: 'Validation failed',
          details: errors.array()
        });
      }

      // Generate job ID
      const jobId = uuidv4();

      // Default configuration
      const defaultConfig = {
        model_name: process.env.DEFAULT_MODEL_NAME || 'openai/gpt-oss-20b',
        datasets_config: [
          {
            source: 'huggingface',
            name: process.env.TRAVEL_CONVERSATIONS_DATASET || 'soniawmeyer/travel-conversations-finetuning',
            file: process.env.TRAVEL_CONVERSATIONS_FILE || 'conversational_sample_processed_with_topic.csv',
            format: 'csv',
            train_split: 0.9,
            priority: 1,
            description: 'Travel Conversations Dataset (680MB)'
          },
          {
            source: 'huggingface',
            name: process.env.TRAVEL_QA_DATASET || 'soniawmeyer/travel-conversations-finetuning',
            file: process.env.TRAVEL_QA_FILE || 'travel_QA_processed_with_topic.csv',
            format: 'csv',
            train_split: 0.9,
            priority: 2,
            description: 'Travel QA Dataset (147MB)'
          }
        ],
        training_config: {
          max_steps_per_dataset: parseInt(process.env.DEFAULT_MAX_STEPS) || 1000,
          learning_rate: parseFloat(process.env.DEFAULT_LEARNING_RATE) || 0.0002,
          batch_size: parseInt(process.env.DEFAULT_BATCH_SIZE) || 4,
          gradient_accumulation_steps: 8,
          warmup_steps: 100,
          save_checkpoint_between_datasets: true,
          fp16: true,
          gradient_checkpointing: true
        },
        lora_config: {
          rank: parseInt(process.env.DEFAULT_LORA_RANK) || 16,
          alpha: parseInt(process.env.DEFAULT_LORA_ALPHA) || 32,
          dropout: 0.1,
          target_modules: [
            'q_proj', 'k_proj', 'v_proj', 'o_proj',
            'gate_proj', 'up_proj', 'down_proj'
          ]
        },
        gpu_config: {
          type: process.env.DEFAULT_GPU_TYPE || 'H100',
          count: 1,
          timeout_hours: parseInt(process.env.DEFAULT_TIMEOUT_HOURS) || 12
        },
        queue_config: {
          process_sequentially: true,
          continue_on_failure: false,
          merge_adapters: true
        }
      };

      // Merge with user configuration
      const jobConfig = {
        ...defaultConfig,
        ...req.body,
        training_config: {
          ...defaultConfig.training_config,
          ...(req.body.training_config || {})
        },
        lora_config: {
          ...defaultConfig.lora_config,
          ...(req.body.lora_config || {})
        },
        gpu_config: {
          ...defaultConfig.gpu_config,
          ...(req.body.gpu_config || {})
        },
        queue_config: {
          ...defaultConfig.queue_config,
          ...(req.body.queue_config || {})
        }
      };

      // Sort datasets by priority
      jobConfig.datasets_config.sort((a, b) => (a.priority || 0) - (b.priority || 0));

      // Create job
      const job = await jobManager.createJob(jobId, jobConfig);

      // Add to queue for sequential processing
      await queueManager.addJob(job);

      // Estimate duration (rough calculation)
      const estimatedDurationPerDataset = 2 + (jobConfig.training_config.max_steps_per_dataset / 500); // hours
      const totalEstimatedDuration = estimatedDurationPerDataset * jobConfig.datasets_config.length;

      // Response with job details
      const response = {
        job_id: jobId,
        status: 'queued',
        estimated_duration: `${Math.ceil(totalEstimatedDuration)}-${Math.ceil(totalEstimatedDuration * 1.5)} hours`,
        total_datasets: jobConfig.datasets_config.length,
        datasets_queue: jobConfig.datasets_config.map((dataset, index) => ({
          dataset_id: `dataset_${index + 1}`,
          description: dataset.description,
          status: 'queued',
          priority: dataset.priority || index + 1
        })),
        created_at: new Date().toISOString(),
        config: jobConfig
      };

      res.status(201).json(response);

      // Start processing if queue was empty
      queueManager.processQueue();

    } catch (error) {
      console.error('Error creating fine-tuning job:', error);
      res.status(500).json({
        error: 'Failed to create fine-tuning job',
        message: error.message
      });
    }
  });

  // GET /api/jobs/:jobId - Get job status
  router.get('/:jobId', validateJobId, async (req, res) => {
    try {
      const errors = validationResult(req);
      if (!errors.isEmpty()) {
        return res.status(400).json({
          error: 'Invalid job ID',
          details: errors.array()
        });
      }

      const { jobId } = req.params;
      const job = await jobManager.getJob(jobId);

      if (!job) {
        return res.status(404).json({
          error: 'Job not found',
          job_id: jobId
        });
      }

      // Get detailed progress information
      const progress = await jobManager.getJobProgress(jobId);
      const queuePosition = await queueManager.getJobPosition(jobId);

      const response = {
        job_id: jobId,
        status: job.status,
        current_dataset: progress.current_dataset,
        overall_progress: {
          datasets_completed: progress.datasets_completed,
          total_datasets: job.config.datasets_config.length,
          overall_percentage: progress.overall_percentage,
          elapsed_time: progress.elapsed_time,
          eta: progress.eta
        },
        current_dataset_progress: progress.current_dataset_progress,
        metrics: progress.metrics,
        datasets_status: progress.datasets_status,
        queue_position: queuePosition,
        logs_url: `/api/jobs/${jobId}/logs`,
        created_at: job.created_at,
        updated_at: job.updated_at
      };

      res.json(response);

    } catch (error) {
      console.error('Error getting job status:', error);
      res.status(500).json({
        error: 'Failed to get job status',
        message: error.message
      });
    }
  });

  // DELETE /api/jobs/:jobId - Cancel job
  router.delete('/:jobId', validateJobId, async (req, res) => {
    try {
      const errors = validationResult(req);
      if (!errors.isEmpty()) {
        return res.status(400).json({
          error: 'Invalid job ID',
          details: errors.array()
        });
      }

      const { jobId } = req.params;
      const result = await jobManager.cancelJob(jobId);

      if (!result.success) {
        return res.status(404).json({
          error: 'Job not found or cannot be cancelled',
          job_id: jobId
        });
      }

      // Remove from queue if still queued
      await queueManager.removeJob(jobId);

      // Notify WebSocket clients
      wsServer.broadcastToJob(jobId, {
        type: 'job_cancelled',
        job_id: jobId,
        timestamp: new Date().toISOString()
      });

      res.json({
        message: 'Job cancelled successfully',
        job_id: jobId,
        status: 'cancelled',
        timestamp: new Date().toISOString()
      });

    } catch (error) {
      console.error('Error cancelling job:', error);
      res.status(500).json({
        error: 'Failed to cancel job',
        message: error.message
      });
    }
  });

  // GET /api/jobs - List all jobs
  router.get('/', async (req, res) => {
    try {
      const { status, limit = 50, offset = 0 } = req.query;
      
      const jobs = await jobManager.listJobs({
        status,
        limit: parseInt(limit),
        offset: parseInt(offset)
      });

      const response = {
        jobs: jobs.map(job => ({
          job_id: job.id,
          status: job.status,
          model_name: job.config.model_name,
          total_datasets: job.config.datasets_config.length,
          created_at: job.created_at,
          updated_at: job.updated_at,
          estimated_duration: job.estimated_duration
        })),
        total: jobs.length,
        limit: parseInt(limit),
        offset: parseInt(offset)
      };

      res.json(response);

    } catch (error) {
      console.error('Error listing jobs:', error);
      res.status(500).json({
        error: 'Failed to list jobs',
        message: error.message
      });
    }
  });

  // GET /api/jobs/:jobId/logs - Get job logs
  router.get('/:jobId/logs', validateJobId, async (req, res) => {
    try {
      const errors = validationResult(req);
      if (!errors.isEmpty()) {
        return res.status(400).json({
          error: 'Invalid job ID',
          details: errors.array()
        });
      }

      const { jobId } = req.params;
      const { lines = 100, level = 'info' } = req.query;

      const logs = await jobManager.getJobLogs(jobId, {
        lines: parseInt(lines),
        level
      });

      if (!logs) {
        return res.status(404).json({
          error: 'Job not found or no logs available',
          job_id: jobId
        });
      }

      res.json({
        job_id: jobId,
        logs: logs,
        timestamp: new Date().toISOString()
      });

    } catch (error) {
      console.error('Error getting job logs:', error);
      res.status(500).json({
        error: 'Failed to get job logs',
        message: error.message
      });
    }
  });

  return router;
};