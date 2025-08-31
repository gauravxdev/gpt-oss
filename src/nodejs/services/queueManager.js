// Queue Manager - Sequential processing of fine-tuning jobs and datasets
const axios = require('axios');
const EventEmitter = require('events');

class QueueManager extends EventEmitter {
  constructor(logger) {
    super();
    this.logger = logger;
    this.jobQueue = new Map(); // jobId -> job
    this.datasetQueue = new Map(); // jobId -> datasets array
    this.processingJob = null;
    this.processingDataset = null;
    this.isProcessing = false;
    this.retryAttempts = new Map(); // jobId -> attempts count
    this.maxRetries = 3;
    
    // Statistics
    this.stats = {
      totalJobsProcessed: 0,
      totalDatasetsProcessed: 0,
      totalFailures: 0,
      averageProcessingTime: 0,
      startTime: Date.now()
    };
  }

  /**
   * Add a job to the queue for sequential processing
   */
  async addJob(job) {
    try {
      this.logger.info(`Adding job to queue: ${job.id}`);
      
      // Add job to main queue
      this.jobQueue.set(job.id, {
        ...job,
        status: 'queued',
        queuedAt: new Date(),
        position: this.jobQueue.size + 1
      });

      // Initialize dataset queue for this job
      this.datasetQueue.set(job.id, job.config.datasets_config.map((dataset, index) => ({
        ...dataset,
        datasetIndex: index,
        status: 'queued',
        jobId: job.id
      })));

      // Initialize retry counter
      this.retryAttempts.set(job.id, 0);

      this.logger.info(`Job ${job.id} added to queue at position ${this.jobQueue.size}`);
      
      // Emit queue update event
      this.emit('queueUpdated', {
        action: 'jobAdded',
        jobId: job.id,
        queueSize: this.jobQueue.size
      });

      return {
        success: true,
        position: this.jobQueue.size,
        estimatedWaitTime: this.calculateEstimatedWaitTime()
      };

    } catch (error) {
      this.logger.error(`Error adding job to queue: ${error.message}`);
      throw error;
    }
  }

  /**
   * Remove a job from the queue
   */
  async removeJob(jobId) {
    try {
      if (this.processingJob && this.processingJob.id === jobId) {
        // Cannot remove currently processing job
        return { success: false, reason: 'Job is currently being processed' };
      }

      const removed = this.jobQueue.delete(jobId);
      this.datasetQueue.delete(jobId);
      this.retryAttempts.delete(jobId);

      if (removed) {
        this.logger.info(`Job ${jobId} removed from queue`);
        this.emit('queueUpdated', {
          action: 'jobRemoved',
          jobId: jobId,
          queueSize: this.jobQueue.size
        });
      }

      return { success: removed };

    } catch (error) {
      this.logger.error(`Error removing job from queue: ${error.message}`);
      throw error;
    }
  }

  /**
   * Get job position in queue
   */
  async getJobPosition(jobId) {
    if (this.processingJob && this.processingJob.id === jobId) {
      return 0; // Currently processing
    }

    const jobIds = Array.from(this.jobQueue.keys());
    const position = jobIds.indexOf(jobId);
    return position === -1 ? null : position + 1;
  }

  /**
   * Start processing the queue
   */
  async processQueue() {
    if (this.isProcessing) {
      this.logger.debug('Queue processing already in progress');
      return;
    }

    this.isProcessing = true;
    this.logger.info('Starting queue processing');

    try {
      while (this.jobQueue.size > 0) {
        // Get next job in queue
        const [jobId, job] = this.jobQueue.entries().next().value;
        
        this.logger.info(`Processing job: ${jobId}`);
        this.processingJob = job;

        // Update job status
        job.status = 'processing';
        job.startedAt = new Date();

        this.emit('jobStarted', {
          jobId: jobId,
          job: job
        });

        // Process all datasets for this job sequentially
        const success = await this.processJobDatasets(job);

        if (success) {
          // Job completed successfully
          job.status = 'completed';
          job.completedAt = new Date();
          this.stats.totalJobsProcessed++;

          this.logger.info(`Job ${jobId} completed successfully`);
          this.emit('jobCompleted', {
            jobId: jobId,
            job: job
          });

        } else {
          // Job failed
          job.status = 'failed';
          job.failedAt = new Date();
          this.stats.totalFailures++;

          this.logger.error(`Job ${jobId} failed`);
          this.emit('jobFailed', {
            jobId: jobId,
            job: job
          });
        }

        // Remove job from queue
        this.jobQueue.delete(jobId);
        this.datasetQueue.delete(jobId);
        this.retryAttempts.delete(jobId);

        this.processingJob = null;
        this.processingDataset = null;
      }

    } catch (error) {
      this.logger.error(`Error in queue processing: ${error.message}`);
      this.emit('queueError', { error });

    } finally {
      this.isProcessing = false;
      this.logger.info('Queue processing completed');
    }
  }

  /**
   * Process all datasets for a job sequentially
   */
  async processJobDatasets(job) {
    try {
      const datasets = this.datasetQueue.get(job.id);
      if (!datasets || datasets.length === 0) {
        this.logger.warn(`No datasets found for job ${job.id}`);
        return false;
      }

      this.logger.info(`Processing ${datasets.length} datasets for job ${job.id}`);

      let allDatasetsSuccessful = true;
      const results = [];

      for (let i = 0; i < datasets.length; i++) {
        const dataset = datasets[i];
        this.processingDataset = dataset;

        this.logger.info(`Processing dataset ${i + 1}/${datasets.length}: ${dataset.description}`);

        // Update dataset status
        dataset.status = 'processing';
        dataset.startedAt = new Date();

        this.emit('datasetStarted', {
          jobId: job.id,
          datasetIndex: i,
          dataset: dataset,
          progress: {
            currentDataset: i + 1,
            totalDatasets: datasets.length
          }
        });

        try {
          // Step 1: Download and validate dataset
          const datasetResult = await this.downloadDataset(job, dataset);
          if (!datasetResult.success) {
            throw new Error(`Dataset download failed: ${datasetResult.error}`);
          }

          // Step 2: Preprocess dataset
          const preprocessResult = await this.preprocessDataset(job, dataset, datasetResult.datasetPath);
          if (!preprocessResult.success) {
            throw new Error(`Dataset preprocessing failed: ${preprocessResult.error}`);
          }

          // Step 3: Fine-tune on dataset
          const finetuneResult = await this.finetuneOnDataset(job, dataset, preprocessResult.processedPath);
          if (!finetuneResult.success) {
            throw new Error(`Fine-tuning failed: ${finetuneResult.error}`);
          }

          // Dataset completed successfully
          dataset.status = 'completed';
          dataset.completedAt = new Date();
          dataset.results = finetuneResult;

          results.push(finetuneResult);
          this.stats.totalDatasetsProcessed++;

          this.logger.info(`Dataset ${i + 1} completed successfully`);
          this.emit('datasetCompleted', {
            jobId: job.id,
            datasetIndex: i,
            dataset: dataset,
            results: finetuneResult
          });

        } catch (error) {
          this.logger.error(`Error processing dataset ${i + 1}: ${error.message}`);

          dataset.status = 'failed';
          dataset.failedAt = new Date();
          dataset.error = error.message;

          this.emit('datasetFailed', {
            jobId: job.id,
            datasetIndex: i,
            dataset: dataset,
            error: error.message
          });

          // Check if we should continue with next dataset
          if (!job.config.queue_config.continue_on_failure) {
            allDatasetsSuccessful = false;
            break;
          }

          allDatasetsSuccessful = false;
        }
      }

      // If requested, merge LoRA adapters from all successful datasets
      if (allDatasetsSuccessful && job.config.queue_config.merge_adapters && results.length > 1) {
        try {
          const mergeResult = await this.mergeLoRAAdapters(job, results);
          if (mergeResult.success) {
            job.finalModel = mergeResult;
            this.logger.info(`LoRA adapters merged successfully for job ${job.id}`);
          }
        } catch (error) {
          this.logger.error(`Error merging LoRA adapters: ${error.message}`);
        }
      }

      return allDatasetsSuccessful || (job.config.queue_config.continue_on_failure && results.length > 0);

    } catch (error) {
      this.logger.error(`Error processing job datasets: ${error.message}`);
      return false;
    }
  }

  /**
   * Download dataset using Modal function
   */
  async downloadDataset(job, dataset) {
    try {
      this.logger.info(`Downloading dataset: ${dataset.name}`);

      // Call Modal dataset loader function
      const response = await axios.post(`${process.env.MODAL_API_URL}/load-dataset-for-training`, {
        dataset_config: dataset,
        job_id: job.id,
        dataset_index: dataset.datasetIndex
      }, {
        timeout: 1800000, // 30 minutes
        headers: {
          'Authorization': `Bearer ${process.env.MODAL_TOKEN}`,
          'Content-Type': 'application/json'
        }
      });

      if (response.data.status === 'ready_for_training') {
        return {
          success: true,
          datasetPath: response.data.cache_path,
          metadata: response.data
        };
      } else {
        return {
          success: false,
          error: response.data.error || 'Unknown error during dataset loading'
        };
      }

    } catch (error) {
      this.logger.error(`Error downloading dataset: ${error.message}`);
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Preprocess dataset using Modal function
   */
  async preprocessDataset(job, dataset, datasetPath) {
    try {
      this.logger.info(`Preprocessing dataset: ${dataset.name}`);

      // Determine preprocessing function based on dataset type
      const isConversationDataset = dataset.file.toLowerCase().includes('conversation');
      const preprocessFunction = isConversationDataset ? 
        'preprocess-conversation-dataset' : 
        'preprocess-qa-dataset';

      // Call Modal preprocessing function
      const response = await axios.post(`${process.env.MODAL_API_URL}/${preprocessFunction}`, {
        dataset_path: datasetPath,
        dataset_config: dataset,
        tokenizer_name: job.config.model_name,
        max_length: job.config.training_config.max_seq_length || 2048,
        job_id: job.id
      }, {
        timeout: 3600000, // 1 hour
        headers: {
          'Authorization': `Bearer ${process.env.MODAL_TOKEN}`,
          'Content-Type': 'application/json'
        }
      });

      if (response.data.status === 'success') {
        return {
          success: true,
          processedPath: response.data.output_path,
          statistics: response.data.statistics,
          sampleData: response.data.sample_data
        };
      } else {
        return {
          success: false,
          error: response.data.error || 'Unknown error during preprocessing'
        };
      }

    } catch (error) {
      this.logger.error(`Error preprocessing dataset: ${error.message}`);
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Fine-tune model on dataset using Modal function
   */
  async finetuneOnDataset(job, dataset, processedPath) {
    try {
      this.logger.info(`Starting fine-tuning on dataset: ${dataset.name}`);

      // Progress callback URL for real-time updates
      const progressCallbackUrl = `${process.env.APP_URL}/api/jobs/${job.id}/progress`;

      // Call Modal fine-tuning function
      const response = await axios.post(`${process.env.MODAL_API_URL}/finetune-gpt-oss-with-lora`, {
        dataset_path: processedPath,
        model_config: {
          model_name: job.config.model_name,
          max_seq_length: job.config.training_config.max_seq_length || 2048
        },
        training_config: job.config.training_config,
        lora_config: job.config.lora_config,
        job_id: job.id,
        dataset_index: dataset.datasetIndex,
        progress_callback: progressCallbackUrl
      }, {
        timeout: job.config.gpu_config.timeout_hours * 3600000, // Convert hours to milliseconds
        headers: {
          'Authorization': `Bearer ${process.env.MODAL_TOKEN}`,
          'Content-Type': 'application/json'
        }
      });

      if (response.data.status === 'completed') {
        return {
          success: true,
          results: response.data
        };
      } else {
        return {
          success: false,
          error: response.data.error || 'Unknown error during fine-tuning'
        };
      }

    } catch (error) {
      this.logger.error(`Error fine-tuning on dataset: ${error.message}`);
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Merge LoRA adapters from multiple datasets
   */
  async mergeLoRAAdapters(job, results) {
    try {
      this.logger.info(`Merging LoRA adapters for job: ${job.id}`);

      const adapterPaths = results.map(result => result.results.adapter_path);

      // Call Modal adapter merging function
      const response = await axios.post(`${process.env.MODAL_API_URL}/merge-lora-adapters`, {
        adapter_paths: adapterPaths,
        base_model: job.config.model_name,
        job_id: job.id,
        output_name: `${job.id}_merged_model`
      }, {
        timeout: 1800000, // 30 minutes
        headers: {
          'Authorization': `Bearer ${process.env.MODAL_TOKEN}`,
          'Content-Type': 'application/json'
        }
      });

      return {
        success: response.data.status === 'success',
        mergedModelPath: response.data.output_path,
        metadata: response.data
      };

    } catch (error) {
      this.logger.error(`Error merging LoRA adapters: ${error.message}`);
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Calculate estimated wait time for queue
   */
  calculateEstimatedWaitTime() {
    if (this.jobQueue.size === 0) {
      return 0;
    }

    // Rough estimation: 4-8 hours per job on average
    const avgJobTime = 6 * 60 * 60 * 1000; // 6 hours in milliseconds
    return this.jobQueue.size * avgJobTime;
  }

  /**
   * Get queue statistics
   */
  getQueueStats() {
    const uptime = Date.now() - this.stats.startTime;
    
    return {
      ...this.stats,
      currentQueueSize: this.jobQueue.size,
      isProcessing: this.isProcessing,
      processingJob: this.processingJob ? {
        id: this.processingJob.id,
        status: this.processingJob.status,
        startedAt: this.processingJob.startedAt
      } : null,
      processingDataset: this.processingDataset ? {
        name: this.processingDataset.name,
        status: this.processingDataset.status,
        startedAt: this.processingDataset.startedAt
      } : null,
      uptime: uptime,
      estimatedWaitTime: this.calculateEstimatedWaitTime()
    };
  }

  /**
   * Get current queue status
   */
  getQueueStatus() {
    const jobs = Array.from(this.jobQueue.values()).map((job, index) => ({
      jobId: job.id,
      position: index + 1,
      status: job.status,
      queuedAt: job.queuedAt,
      estimatedStartTime: new Date(Date.now() + (index * 6 * 60 * 60 * 1000))
    }));

    return {
      queueSize: this.jobQueue.size,
      isProcessing: this.isProcessing,
      jobs: jobs,
      processingJob: this.processingJob,
      processingDataset: this.processingDataset,
      stats: this.getQueueStats()
    };
  }
}

module.exports = QueueManager;