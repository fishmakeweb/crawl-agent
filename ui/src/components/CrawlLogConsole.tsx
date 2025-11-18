// Real-time crawl log console component
import React, { useState, useEffect, useRef } from 'react';
import wsService from '../services/websocket';
import type { CrawlLogEntry } from '../types';
import './CrawlLogConsole.css';

interface CrawlLogConsoleProps {
  jobId: string | null;
  isActive: boolean;
}

export const CrawlLogConsole: React.FC<CrawlLogConsoleProps> = ({ jobId, isActive }) => {
  const [logs, setLogs] = useState<CrawlLogEntry[]>([]);
  const [isCollapsed, setIsCollapsed] = useState(false);
  const logEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleLog = (data: any) => {
      const log: CrawlLogEntry = {
        level: data.level || 'INFO',
        message: data.message || '',
        logger: data.logger || '',
        timestamp: data.timestamp || new Date().toISOString(),
        job_id: data.job_id || null,
      };

      // Only add logs matching current job_id (or all if jobId is null)
      if (jobId === null || log.job_id === jobId) {
        setLogs((prev) => [...prev, log]);
      }
    };

    wsService.on('crawl_log', handleLog);

    return () => {
      wsService.off('crawl_log', handleLog);
    };
  }, [jobId]);

  // Auto-scroll to bottom when new logs arrive
  useEffect(() => {
    if (!isCollapsed && logEndRef.current) {
      logEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs, isCollapsed]);

  // Clear logs when starting new crawl
  useEffect(() => {
    if (isActive && jobId) {
      setLogs([]);
    }
  }, [jobId, isActive]);

  const handleClear = () => {
    setLogs([]);
  };

  const toggleCollapse = () => {
    setIsCollapsed(!isCollapsed);
  };

  // Extract log type from message ([INIT], [PLAN], etc.)
  const getLogType = (message: string): string => {
    const match = message.match(/\[(\w+)\]/);
    return match ? match[1].toLowerCase() : 'info';
  };

  // Format timestamp to HH:MM:SS.mmm
  const formatTime = (timestamp: string): string => {
    try {
      const date = new Date(timestamp);
      const hours = date.getHours().toString().padStart(2, '0');
      const minutes = date.getMinutes().toString().padStart(2, '0');
      const seconds = date.getSeconds().toString().padStart(2, '0');
      const ms = date.getMilliseconds().toString().padStart(3, '0');
      return `${hours}:${minutes}:${seconds}.${ms}`;
    } catch {
      return timestamp;
    }
  };

  if (!isActive && logs.length === 0) {
    return null; // Don't show console if not active and no logs
  }

  return (
    <div className={`log-console ${isCollapsed ? 'collapsed' : 'expanded'}`}>
      <div className="log-header" onClick={toggleCollapse}>
        <span className="log-title">
          {isCollapsed ? '▶' : '▼'} 🔍 Crawl Logs {logs.length > 0 && `(${logs.length})`}
        </span>
        <div className="log-controls" onClick={(e) => e.stopPropagation()}>
          {!isCollapsed && (
            <button className="log-btn" onClick={handleClear} title="Clear logs">
              Clear
            </button>
          )}
        </div>
      </div>

      {!isCollapsed && (
        <div className="log-content">
          {logs.length === 0 ? (
            <div className="log-empty">No logs yet. Start a crawl to see real-time logs.</div>
          ) : (
            logs.map((log, idx) => (
              <div key={idx} className={`log-entry log-${getLogType(log.message)}`}>
                <span className="log-time">{formatTime(log.timestamp)}</span>
                <span className="log-message">{log.message}</span>
              </div>
            ))
          )}
          <div ref={logEndRef} />
        </div>
      )}
    </div>
  );
};
