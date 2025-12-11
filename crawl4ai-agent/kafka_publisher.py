"""
Kafka Progress Publisher for Crawl4AI Agent
Publishes real-time crawl progress events to Kafka topics
"""
import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from confluent_kafka import Producer
from confluent_kafka.error import KafkaException

logger = logging.getLogger(__name__)


class Crawl4AIKafkaPublisher:
    """
    Publishes Crawl4AI progress events to Kafka for real-time monitoring
    """

    def __init__(self, bootstrap_servers: Optional[str] = None):
        """
        Initialize Kafka producer

        Args:
            bootstrap_servers: Kafka bootstrap servers (default: from env or localhost:9092)
        """
        self.bootstrap_servers = bootstrap_servers or os.getenv(
            'KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092'
        )
        self.client_id = os.getenv('KAFKA_CLIENT_ID', 'crawl4ai-agent')
        self.producer: Optional[Producer] = None
        self.enabled = os.getenv('KAFKA_ENABLED', 'true').lower() == 'true'

        if self.enabled:
            logger.info(
                "Kafka publishing enabled (client_id=%s, bootstrap=%s)",
                self.client_id,
                self.bootstrap_servers
            )
            self._initialize_producer()
        else:
            logger.info("Kafka publishing is disabled (KAFKA_ENABLED=false)")

    def _initialize_producer(self):
        """Initialize Kafka producer with configuration"""
        try:
            config = {
                'bootstrap.servers': self.bootstrap_servers,
                'client.id': self.client_id,
                'acks': 'all',  # Wait for all replicas
                'compression.type': 'snappy',
                'linger.ms': 10,  # Batch messages for 10ms for efficiency
                'batch.size': 16384,  # 16KB batch size
                'max.in.flight.requests.per.connection': 5,
                'enable.idempotence': True,  # Exactly-once semantics
                'retries': 3,
                'retry.backoff.ms': 100,
                'log.connection.close': False,  # Suppress connection close logs
                'log_level': 3  # 3=ERROR (suppress INFO/DEBUG/WARNING from librdkafka)
            }

            self.producer = Producer(config)
            logger.info("Kafka producer initialized against %s", self.bootstrap_servers)

            # Fetch cluster metadata immediately so connection failures surface in logs
            try:
                metadata = self.producer.list_topics(timeout=5.0)
                broker_hosts = ",".join(
                    sorted(f"{b.host}:{b.port}" for b in metadata.brokers.values())
                ) or "unknown"
                logger.info(
                    "Kafka metadata fetched successfully (cluster_id=%s, brokers=%s)",
                    getattr(metadata, 'cluster_id', 'n/a'),
                    broker_hosts
                )
            except KafkaException as meta_ex:
                logger.error(
                    "Kafka metadata request failed (bootstrap=%s): %s",
                    self.bootstrap_servers,
                    meta_ex
                )
                logger.warning(
                    "Producer will stay enabled but Kafka connectivity issues are likely"
                )

        except Exception as ex:
            logger.error(f"Failed to initialize Kafka producer: {ex}")
            logger.warning("Crawl will continue without Kafka progress events")
            self.enabled = False

    def publish_progress(
        self,
        event_type: str,
        job_id: str,
        user_id: str,
        data: Dict[str, Any]
    ) -> bool:
        """
        Publish a progress event to Kafka

        Args:
            event_type: Type of event (e.g., "NavigationPlanningStarted", "PaginationPageLoaded")
            job_id: Crawl job ID (GUID from C#)
            user_id: User ID (GUID from C#)
            data: Event-specific data payload

        Returns:
            True if published successfully, False otherwise
        """
        if not self.enabled or not self.producer:
            return False

        try:
            topic = 'crawler.job.progress'

            event = {
                "eventType": event_type,
                "jobId": job_id,
                "userId": user_id,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "data": data
            }

            # Serialize to JSON with camelCase (matching C# conventions)
            message_value = json.dumps(event, default=str)

            # Produce message asynchronously
            self.producer.produce(
                topic=topic,
                key=job_id,  # Use jobId as key for partition ordering
                value=message_value,
                headers={'service': 'crawl4ai-agent'},
                callback=self._delivery_callback
            )

            # Trigger delivery reports (non-blocking)
            self.producer.poll(0)

            logger.debug(f"Published event: {event_type} for job {job_id}")
            return True

        except BufferError:
            logger.warning(f"Kafka producer queue is full, will retry: {event_type}")
            self.producer.poll(0.1)  # Wait for space
            return False

        except Exception as ex:
            logger.error(f"Failed to publish event {event_type}: {ex}")
            return False

    def publish_error(self, job_id: str, user_id: str, error_message: str, error_details: Optional[Dict] = None):
        """
        Publish an error event

        Args:
            job_id: Crawl job ID
            user_id: User ID
            error_message: Error message
            error_details: Optional additional error details
        """
        data = {
            "errorMessage": error_message,
            "errorDetails": error_details or {},
            "failedAt": datetime.utcnow().isoformat() + "Z"
        }

        self.publish_progress("CrawlError", job_id, user_id, data)

    def _delivery_callback(self, err, msg):
        """
        Callback for Kafka message delivery reports

        Args:
            err: Error if delivery failed
            msg: Message metadata
        """
        if err:
            logger.error(f"Message delivery failed: {err}")
        else:
            logger.debug(
                f"Message delivered to {msg.topic()} [partition {msg.partition()}] "
                f"at offset {msg.offset()}"
            )

    def flush(self, timeout: float = 10.0):
        """
        Flush pending messages

        Args:
            timeout: Maximum time to wait in seconds
        """
        if self.producer:
            remaining = self.producer.flush(timeout)
            if remaining > 0:
                logger.warning(f"{remaining} messages not delivered before timeout")

    def close(self):
        """Close the Kafka producer"""
        if self.producer:
            self.flush()
            logger.info("Kafka producer closed")


# Example usage for testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    publisher = Crawl4AIKafkaPublisher()

    test_job_id = "3fa85f64-5717-4562-b3fc-2c963f66afa6"
    test_user_id = "user-guid-1234"

    # Test event publishing
    publisher.publish_progress(
        "NavigationPlanningStarted",
        test_job_id,
        test_user_id,
        {
            "url": "https://example.com",
            "prompt": "Extract product information"
        }
    )

    publisher.publish_progress(
        "PaginationPageLoaded",
        test_job_id,
        test_user_id,
        {
            "url": "https://example.com/page/2",
            "currentPage": 2,
            "totalPagesProcessed": 2,
            "maxPages": 50,
            "itemsExtractedSoFar": 45,
            "progressPercentage": 40.0
        }
    )

    publisher.close()
