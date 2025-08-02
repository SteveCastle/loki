# Shrike HTTP API Documentation

Shrike is a job management system with web UI that handles media processing tasks and provides real-time streaming updates via Server-Sent Events (SSE).

## Server Information

- **Base URL**: `http://localhost:8090`
- **Protocol**: HTTP
- **Port**: 8090 (configurable in code)

## Available Tasks

The system supports the following built-in tasks:

| Task ID | Name | Description |
|---------|------|-------------|
| `wait` | Wait | Test task that waits 5 seconds then completes |
| `gallery-dl` | gallery-dl | Download media from websites using gallery-dl |
| `dce` | dce | Execute dce command |
| `yt-dlp` | yt-dlp | Download videos using yt-dlp |
| `ffmpeg` | ffmpeg | Process media files using ffmpeg |
| `remove` | Remove Media | Remove media items from database |
| `cleanup` | CleanUp | Remove media items from database that no longer exist in filesystem |
| `ingest` | Ingest Media Files | Scan directories and add media files to database |
| `metadata` | Generate Metadata | Generate descriptions, transcripts, hashes, and dimensions for media files |
| `move` | Move Media Files | Move media files to new location while updating database references |

## API Endpoints

### Home Page & Job List
- **GET** `/`
  - Returns HTML page with list of all jobs
  - Shows job status, timestamps, and controls

### Job Creation

#### Create Job (New API)
- **POST** `/create`
- **Content-Type**: `application/json`
- **Request Body**:
  ```json
  {
    "input": "task_name arg1 arg2 input_value"
  }
  ```
- **Response**: 
  ```json
  {
    "id": "uuid-string"
  }
  ```
- **Example**:
  ```bash
  curl -X POST http://localhost:8090/create \
    -H "Content-Type: application/json" \
    -d '{"input": "yt-dlp --format best https://youtube.com/watch?v=example"}'
  ```

#### Create Job (Legacy API)
- **POST** `/`
- **Content-Type**: `application/json`
- **Request Body**:
  ```json
  {
    "command": "task_name",
    "arguments": ["arg1", "arg2", "input_value"]
  }
  ```
- **Response**: 
  ```json
  {
    "id": "uuid-string"
  }
  ```

### Job Management

#### View Job Details
- **GET** `/job/{id}`
- Returns detailed HTML view of specific job including:
  - Job metadata (ID, command, arguments, state)
  - Real-time stdout output
  - Timestamps (created, claimed, completed/errored)
  - Action buttons (cancel, copy, remove)

#### Cancel Job
- **POST** `/job/{id}/cancel`
- Cancels a pending or in-progress job
- **Response**: `200 OK` with message "Job cancelled successfully"

#### Copy Job
- **POST** `/job/{id}/copy`
- Creates a new job with same parameters as existing job
- **Response**: 
  ```json
  {
    "id": "new-uuid",
    "message": "Job copied successfully"
  }
  ```

#### Remove Job
- **POST** `/job/{id}/remove`
- Removes job from queue and database
- **Response**: `200 OK` with message "Job removed successfully"

#### Clear Non-Running Jobs
- **POST** `/jobs/clear`
- Removes all jobs except those currently in progress
- **Response**: 
  ```json
  {
    "cleared_count": 5,
    "message": "Cleared 5 non-running jobs"
  }
  ```

### System Information

#### Health Check
- **GET** `/health`
- **Response**:
  ```json
  {
    "status": "healthy",
    "timestamp": 1704067200,
    "stream": {
      "active_connections": 2,
      "total_messages": 150,
      "max_connections": 1000
    },
    "jobs": {
      "total": 10,
      "pending": 2,
      "in_progress": 1,
      "completed": 5,
      "cancelled": 1,
      "error": 1
    }
  }
  ```

### Media Browser

#### Media Gallery Page
- **GET** `/media`
- **Query Parameters**:
  - `q`: Search query (optional)
- Returns HTML page with media gallery interface

#### Media API (JSON)
- **GET** `/media/api`
- **Query Parameters**:
  - `offset`: Starting offset for pagination (default: 0)
  - `limit`: Number of items to return (default: 25, max: 100)
  - `q`: Search query (optional)
  - `path`: Specific file path for single item lookup
  - `single`: Set to "true" for single item lookup by path
- **Response**:
  ```json
  {
    "items": [
      {
        "path": "/path/to/media/file.jpg",
        "description": "Generated description",
        "hash": "sha256hash",
        "size": 1024000,
        "width": 1920,
        "height": 1080
      }
    ],
    "hasMore": true
  }
  ```

#### Media File Serving
- **GET** `/media/file`
- **Query Parameters**:
  - `path`: URL-encoded path to media file
- Serves individual media files with appropriate Content-Type headers
- Supports both local files and remote URL proxying
- Includes caching headers (ETag, Cache-Control)

### Real-Time Updates (Server-Sent Events)

#### SSE Stream Connection
- **GET** `/stream`
- **Response**: `text/event-stream`
- **Headers**:
  - `Content-Type: text/event-stream`
  - `Cache-Control: no-cache`
  - `Connection: keep-alive`

#### SSE Event Types

1. **Connection Events**
   ```
   data: {"type":"connected","msg":"SSE connection established"}
   ```

2. **Job List Updates**
   ```
   event: create
   data: {"updateType":"create","job":{...},"html":"<tr>...</tr>"}
   
   event: update  
   data: {"updateType":"update","job":{...},"html":"<tr>...</tr>"}
   
   event: delete
   data: {"updateType":"delete","job":{"id":"uuid"},"html":""}
   ```

3. **Job Output Updates**
   ```
   event: stdout-{job-id}
   data: {"updateType":"stdout","line":"Output line from job"}
   ```

4. **Keep-Alive**
   ```
   : keep-alive
   ```

## Job Lifecycle & States

### Job States
- **Pending** (0): Job created, waiting for dependencies and available runner
- **InProgress** (1): Job currently being executed  
- **Completed** (2): Job finished successfully
- **Cancelled** (3): Job was cancelled by user
- **Error** (4): Job failed with an error

### Job Processing Flow
1. **Creation**: Job created via POST request, assigned UUID, added to queue
2. **Queuing**: Job waits for dependencies to complete and runner availability
3. **Claiming**: Available runner claims next eligible job (FIFO with dependency resolution)
4. **Execution**: Task function executes with real-time stdout streaming
5. **Completion**: Job marked as completed/error, final state saved to database

### Dependencies
- Jobs can specify dependency IDs in the `dependencies` array
- Jobs only start after ALL dependencies are completed
- Failed dependencies prevent dependent jobs from starting

## Detailed Task Examples

### Media Processing Tasks

#### Gallery-dl Download
```bash
curl -X POST http://localhost:8090/create \
  -H "Content-Type: application/json" \
  -d '{"input": "gallery-dl --dest /downloads https://example.com/gallery"}'
```

#### Video Download with yt-dlp
```bash
curl -X POST http://localhost:8090/create \
  -H "Content-Type: application/json" \
  -d '{"input": "yt-dlp --format bestvideo+bestaudio --output \"/downloads/%(title)s.%(ext)s\" https://youtube.com/watch?v=VIDEO_ID"}'
```

#### Media File Conversion
```bash
curl -X POST http://localhost:8090/create \
  -H "Content-Type: application/json" \
  -d '{"input": "ffmpeg -i input.mp4 -c:v libx264 -crf 23 output.mp4"}'
```

### Database Management Tasks

#### Ingest Media Files
```bash
curl -X POST http://localhost:8090/create \
  -H "Content-Type: application/json" \
  -d '{"input": "ingest -r /path/to/media/directory"}'
```

#### Generate Metadata
```bash
# Generate all metadata types for all files
curl -X POST http://localhost:8090/create \
  -H "Content-Type: application/json" \
  -d '{"input": "metadata --type description,hash,dimensions --apply all"}'

# Generate only descriptions for specific files
curl -X POST http://localhost:8090/create \
  -H "Content-Type: application/json" \
  -d '{"input": "metadata --type description --model llama3.2-vision\n/path/to/image1.jpg\n/path/to/video1.mp4"}'
```

#### Move Media Files
```bash
curl -X POST http://localhost:8090/create \
  -H "Content-Type: application/json" \
  -d '{"input": "move /new/target/directory --prefix /old/common/path\n/old/path/file1.jpg\n/old/path/file2.mp4"}'
```

#### Database Cleanup
```bash
# Remove orphaned database entries
curl -X POST http://localhost:8090/create \
  -H "Content-Type: application/json" \  
  -d '{"input": "cleanup"}'

# Remove specific files from database
curl -X POST http://localhost:8090/create \
  -H "Content-Type: application/json" \
  -d '{"input": "remove\n/path/to/file1.jpg\n/path/to/file2.mp4"}'
```

## Real-Time Job Monitoring

### Using SSE in JavaScript
```javascript
const eventSource = new EventSource('http://localhost:8090/stream');

// Listen for job updates
eventSource.addEventListener('create', (event) => {
  const data = JSON.parse(event.data);
  console.log('New job created:', data.job.id);
});

eventSource.addEventListener('update', (event) => {
  const data = JSON.parse(event.data);
  console.log('Job updated:', data.job.id, 'State:', data.job.state);
});

// Listen for specific job output
eventSource.addEventListener('stdout-JOB_ID_HERE', (event) => {
  const data = JSON.parse(event.data);
  console.log('Job output:', data.line);
});

eventSource.onerror = (error) => {
  console.error('SSE connection error:', error);
};
```

### Using curl for SSE
```bash
# Monitor all job events
curl -N http://localhost:8090/stream

# Monitor and filter for specific events
curl -N http://localhost:8090/stream | grep "event: stdout-"
```

## Error Handling

### HTTP Status Codes
- **200 OK**: Successful request
- **201 Created**: Job created successfully  
- **400 Bad Request**: Invalid JSON or missing required fields
- **404 Not Found**: Job not found
- **405 Method Not Allowed**: Wrong HTTP method
- **500 Internal Server Error**: Server error during processing
- **503 Service Unavailable**: Server at capacity (SSE connections)

### Error Response Format
```json
{
  "error": "Description of the error",
  "code": "ERROR_CODE"
}
```

## Database Persistence

- All jobs are persisted to SQLite database
- Job state, metadata, and output are automatically saved
- Jobs resume as "Pending" if server restarts while in progress
- Database location configured via `config.json` in `%APPDATA%\Lowkey Media Viewer\`

## Connection Limits & Performance

### SSE Connection Limits
- Maximum concurrent SSE connections: 1000
- Client channel buffer size: 50 messages
- Keep-alive interval: 30 seconds
- Cleanup interval: 60 seconds
- Client send timeout: 5 seconds

### Media File Serving
- Maximum file size for preview: 100MB
- Cache duration: 1 hour for local files, 30 minutes for remote
- Security: Basic directory traversal protection
- Remote URL proxying supported with 30-second timeout

## Configuration

The system reads configuration from:
- **Windows**: `%APPDATA%\Lowkey Media Viewer\config.json`

Required configuration:
```json
{
  "dbPath": "C:\\path\\to\\database.db"
}
```

## System Requirements

- **Windows OS** (system tray integration)
- **SQLite database** (modernc.org/sqlite)
- **Optional external tools**:
  - `gallery-dl` for website downloads
  - `yt-dlp` for video downloads  
  - `ffmpeg`/`ffprobe` for media processing
  - `faster-whisper-xxl` for transcription
  - **Ollama** with vision models for image descriptions

## Security Considerations

- Basic directory traversal protection for file serving
- No authentication/authorization implemented
- Intended for local development/personal use
- Media file access restricted to configured paths
- SSE connections have reasonable limits and timeouts