// WebSocket Service - Real-time updates for fine-tuning progress
const WebSocket = require('ws');
const { v4: uuidv4 } = require('uuid');

class WebSocketServer {
  constructor(logger) {
    this.logger = logger;
    this.wss = null;
    this.clients = new Map(); // clientId -> { ws, jobIds, metadata }
    this.jobSubscriptions = new Map(); // jobId -> Set of clientIds
    this.heartbeatInterval = null;
  }

  /**
   * Start the WebSocket server
   */
  start(port = 3001) {
    this.wss = new WebSocket.Server({ 
      port,
      perMessageDeflate: false 
    });

    this.logger.info(`WebSocket server starting on port ${port}`);

    this.wss.on('connection', (ws, req) => {
      this.handleConnection(ws, req);
    });

    this.wss.on('error', (error) => {
      this.logger.error('WebSocket server error:', error);
    });

    // Start heartbeat interval
    this.startHeartbeat();

    this.logger.info(`WebSocket server started on port ${port}`);
  }

  /**
   * Handle new WebSocket connections
   */
  handleConnection(ws, req) {
    const clientId = uuidv4();
    const clientInfo = {
      ws,
      jobIds: new Set(),
      connectedAt: new Date(),
      lastHeartbeat: Date.now(),
      metadata: {
        userAgent: req.headers['user-agent'],
        ip: req.socket.remoteAddress
      }
    };

    this.clients.set(clientId, clientInfo);
    this.logger.info(`WebSocket client connected: ${clientId}`);

    // Send welcome message
    this.sendToClient(clientId, {
      type: 'connection_established',
      client_id: clientId,
      timestamp: new Date().toISOString(),
      server_info: {
        version: '1.0.0',
        features: ['real_time_progress', 'job_notifications', 'queue_updates']
      }
    });

    // Handle messages from client
    ws.on('message', (data) => {
      this.handleClientMessage(clientId, data);
    });

    // Handle client disconnection
    ws.on('close', (code, reason) => {
      this.handleDisconnection(clientId, code, reason);
    });

    // Handle WebSocket errors
    ws.on('error', (error) => {
      this.logger.error(`WebSocket client ${clientId} error:`, error);
      this.handleDisconnection(clientId, 1006, 'error');
    });

    // Handle pong responses for heartbeat
    ws.on('pong', () => {
      if (this.clients.has(clientId)) {
        this.clients.get(clientId).lastHeartbeat = Date.now();
      }
    });
  }

  /**
   * Handle messages from WebSocket clients
   */
  handleClientMessage(clientId, data) {
    try {
      const message = JSON.parse(data.toString());
      const client = this.clients.get(clientId);

      if (!client) {
        return;
      }

      this.logger.debug(`Message from client ${clientId}:`, message.type);

      switch (message.type) {
        case 'subscribe_job':
          this.subscribeToJob(clientId, message.job_id);
          break;

        case 'unsubscribe_job':
          this.unsubscribeFromJob(clientId, message.job_id);
          break;

        case 'subscribe_queue':
          this.subscribeToQueue(clientId);
          break;

        case 'unsubscribe_queue':
          this.unsubscribeFromQueue(clientId);
          break;

        case 'heartbeat':
          this.sendToClient(clientId, {
            type: 'heartbeat_response',
            timestamp: new Date().toISOString()
          });
          break;

        case 'get_status':
          this.sendConnectionStatus(clientId);
          break;

        default:
          this.logger.warn(`Unknown message type from client ${clientId}: ${message.type}`);
          this.sendToClient(clientId, {
            type: 'error',
            message: 'Unknown message type',
            timestamp: new Date().toISOString()
          });
      }

    } catch (error) {
      this.logger.error(`Error handling client message from ${clientId}:`, error);
      this.sendToClient(clientId, {
        type: 'error',
        message: 'Invalid message format',
        timestamp: new Date().toISOString()
      });
    }
  }

  /**
   * Handle client disconnection
   */
  handleDisconnection(clientId, code, reason) {
    const client = this.clients.get(clientId);
    if (!client) {
      return;
    }

    this.logger.info(`WebSocket client disconnected: ${clientId}, code: ${code}, reason: ${reason}`);

    // Remove from all job subscriptions
    for (const jobId of client.jobIds) {
      if (this.jobSubscriptions.has(jobId)) {
        this.jobSubscriptions.get(jobId).delete(clientId);
        if (this.jobSubscriptions.get(jobId).size === 0) {
          this.jobSubscriptions.delete(jobId);
        }
      }
    }

    // Remove client
    this.clients.delete(clientId);
  }

  /**
   * Subscribe client to job updates
   */
  subscribeToJob(clientId, jobId) {
    const client = this.clients.get(clientId);
    if (!client) {
      return;
    }

    // Add client to job subscription
    if (!this.jobSubscriptions.has(jobId)) {
      this.jobSubscriptions.set(jobId, new Set());
    }
    this.jobSubscriptions.get(jobId).add(clientId);
    client.jobIds.add(jobId);

    this.logger.info(`Client ${clientId} subscribed to job ${jobId}`);

    // Send confirmation
    this.sendToClient(clientId, {
      type: 'subscription_confirmed',
      job_id: jobId,
      timestamp: new Date().toISOString()
    });
  }

  /**
   * Unsubscribe client from job updates
   */
  unsubscribeFromJob(clientId, jobId) {
    const client = this.clients.get(clientId);
    if (!client) {
      return;
    }

    // Remove client from job subscription
    if (this.jobSubscriptions.has(jobId)) {
      this.jobSubscriptions.get(jobId).delete(clientId);
      if (this.jobSubscriptions.get(jobId).size === 0) {
        this.jobSubscriptions.delete(jobId);
      }
    }
    client.jobIds.delete(jobId);

    this.logger.info(`Client ${clientId} unsubscribed from job ${jobId}`);

    // Send confirmation
    this.sendToClient(clientId, {
      type: 'unsubscription_confirmed',
      job_id: jobId,
      timestamp: new Date().toISOString()
    });
  }

  /**
   * Subscribe client to queue updates
   */
  subscribeToQueue(clientId) {
    const client = this.clients.get(clientId);
    if (!client) {
      return;
    }

    client.subscribeToQueue = true;

    this.sendToClient(clientId, {
      type: 'queue_subscription_confirmed',
      timestamp: new Date().toISOString()
    });
  }

  /**
   * Unsubscribe client from queue updates
   */
  unsubscribeFromQueue(clientId) {
    const client = this.clients.get(clientId);
    if (!client) {
      return;
    }

    client.subscribeToQueue = false;

    this.sendToClient(clientId, {
      type: 'queue_unsubscription_confirmed',
      timestamp: new Date().toISOString()
    });
  }

  /**
   * Send message to specific client
   */
  sendToClient(clientId, message) {
    const client = this.clients.get(clientId);
    if (!client || client.ws.readyState !== WebSocket.OPEN) {
      return false;
    }

    try {
      client.ws.send(JSON.stringify(message));
      return true;
    } catch (error) {
      this.logger.error(`Error sending message to client ${clientId}:`, error);
      return false;
    }
  }

  /**
   * Broadcast message to all clients subscribed to a job
   */
  broadcastToJob(jobId, message) {
    const subscribers = this.jobSubscriptions.get(jobId);
    if (!subscribers) {
      return 0;
    }

    let sentCount = 0;
    for (const clientId of subscribers) {
      if (this.sendToClient(clientId, message)) {
        sentCount++;
      }
    }

    this.logger.debug(`Broadcasted message to ${sentCount} clients for job ${jobId}`);
    return sentCount;
  }

  /**
   * Broadcast message to all clients subscribed to queue updates
   */
  broadcastToQueue(message) {
    let sentCount = 0;
    for (const [clientId, client] of this.clients) {
      if (client.subscribeToQueue) {
        if (this.sendToClient(clientId, message)) {
          sentCount++;
        }
      }
    }

    this.logger.debug(`Broadcasted queue message to ${sentCount} clients`);
    return sentCount;
  }

  /**
   * Broadcast message to all connected clients
   */
  broadcastToAll(message) {
    let sentCount = 0;
    for (const clientId of this.clients.keys()) {
      if (this.sendToClient(clientId, message)) {
        sentCount++;
      }
    }

    this.logger.debug(`Broadcasted message to ${sentCount} clients`);
    return sentCount;
  }

  /**
   * Send training progress update
   */
  sendTrainingProgress(jobId, progressData) {
    const message = {
      type: 'training_progress',
      job_id: jobId,
      data: progressData,
      timestamp: new Date().toISOString()
    };

    return this.broadcastToJob(jobId, message);
  }

  /**
   * Send metrics update
   */
  sendMetricsUpdate(jobId, metrics) {
    const message = {
      type: 'metrics_update',
      job_id: jobId,
      metrics: metrics,
      timestamp: new Date().toISOString()
    };

    return this.broadcastToJob(jobId, message);
  }

  /**
   * Send log message
   */
  sendLogMessage(jobId, logData) {
    const message = {
      type: 'log_message',
      job_id: jobId,
      log: logData,
      timestamp: new Date().toISOString()
    };

    return this.broadcastToJob(jobId, message);
  }

  /**
   * Send job status change
   */
  sendJobStatusChange(jobId, oldStatus, newStatus, details = {}) {
    const message = {
      type: 'status_change',
      job_id: jobId,
      old_status: oldStatus,
      new_status: newStatus,
      details: details,
      timestamp: new Date().toISOString()
    };

    return this.broadcastToJob(jobId, message);
  }

  /**
   * Send error notification
   */
  sendErrorNotification(jobId, error) {
    const message = {
      type: 'error_occurred',
      job_id: jobId,
      error: {
        message: error.message,
        stack: error.stack,
        timestamp: new Date().toISOString()
      },
      timestamp: new Date().toISOString()
    };

    return this.broadcastToJob(jobId, message);
  }

  /**
   * Send queue update
   */
  sendQueueUpdate(queueData) {
    const message = {
      type: 'queue_update',
      data: queueData,
      timestamp: new Date().toISOString()
    };

    return this.broadcastToQueue(message);
  }

  /**
   * Send connection status to client
   */
  sendConnectionStatus(clientId) {
    const client = this.clients.get(clientId);
    if (!client) {
      return;
    }

    const status = {
      type: 'connection_status',
      client_id: clientId,
      connected_at: client.connectedAt,
      subscribed_jobs: Array.from(client.jobIds),
      subscribed_to_queue: client.subscribeToQueue || false,
      server_stats: this.getServerStats(),
      timestamp: new Date().toISOString()
    };

    this.sendToClient(clientId, status);
  }

  /**
   * Start heartbeat to detect dead connections
   */
  startHeartbeat() {
    this.heartbeatInterval = setInterval(() => {
      const now = Date.now();
      const timeout = 30000; // 30 seconds

      for (const [clientId, client] of this.clients) {
        if (client.ws.readyState === WebSocket.OPEN) {
          if (now - client.lastHeartbeat > timeout) {
            this.logger.warn(`Client ${clientId} heartbeat timeout, closing connection`);
            client.ws.terminate();
            this.handleDisconnection(clientId, 1001, 'heartbeat_timeout');
          } else {
            // Send ping
            client.ws.ping();
          }
        }
      }
    }, 15000); // Check every 15 seconds
  }

  /**
   * Get server statistics
   */
  getServerStats() {
    return {
      connected_clients: this.clients.size,
      job_subscriptions: this.jobSubscriptions.size,
      total_subscriptions: Array.from(this.jobSubscriptions.values())
        .reduce((total, set) => total + set.size, 0),
      uptime: process.uptime(),
      memory_usage: process.memoryUsage()
    };
  }

  /**
   * Close WebSocket server
   */
  close() {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
    }

    if (this.wss) {
      // Close all client connections
      for (const [clientId, client] of this.clients) {
        client.ws.close(1001, 'server_shutdown');
      }

      this.wss.close(() => {
        this.logger.info('WebSocket server closed');
      });
    }
  }
}

module.exports = WebSocketServer;