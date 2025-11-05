# Docker Deployment Guide

Deploy the luggage monitoring system using Docker Compose locally or on platforms like Coolify.

**Note**: All Docker files are in the `app/` directory.

---

## For Coolify/Remote Servers

If you're deploying on **Coolify** or having network issues during build, use the simpler Dockerfile:

### Option 1: Use Dockerfile.simple (Recommended for Coolify)

In Coolify, set:
- **Dockerfile Location**: `app/Dockerfile.simple`
- **Docker Compose File**: `app/docker-compose.simple.yml`

Or manually build:
```bash
cd app
docker-compose -f docker-compose.simple.yml up -d
```

**Dockerfile.simple uses `python:3.11` (full image) instead of `python:3.11-slim`** which avoids network issues with apt-get on restricted servers.

### Option 2: Use Regular Dockerfile

If Option 1 doesn't work, try the optimized multi-stage build:
```bash
cd app
docker-compose up --build
```

---

## Local Development Prerequisites

### Required Software

1. **Docker Desktop**
   - **Windows**: [Download Docker Desktop for Windows](https://www.docker.com/products/docker-desktop)
   - **macOS**: [Download Docker Desktop for Mac](https://www.docker.com/products/docker-desktop)
   - **Linux**: [Install Docker Engine](https://docs.docker.com/engine/install/)

2. **Verify Installation**
   ```bash
   docker --version
   docker-compose --version
   ```

   **Expected output:**
   ```
   Docker version 24.x.x
   Docker Compose version v2.x.x
   ```

---

## Quick Start (5 Minutes)

### Step 0: Navigate to App Directory

```bash
cd app
```

### Step 1: Configure Environment

Create `.env.docker` file (already created):

```bash
# Copy and edit with your values
cp .env.docker .env.docker.local

# Edit with your actual API keys
notepad .env.docker.local  # Windows
nano .env.docker.local     # Linux/Mac
```

**Required values:**
```env
GOOGLE_API_KEY=your_actual_gemini_api_key
EMAIL_USER=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
EMAIL_FROM=your_email@gmail.com
```

### Step 2: Build and Start Services

```bash
# From project root directory
cd C:\Files\Projects\CCTV

# Build and start all services
docker-compose --env-file .env.docker.local up --build
```

**What happens:**
1. ✅ Builds Docker image (~5-10 minutes first time)
2. ✅ Starts MySQL database
3. ✅ Waits for MySQL to be healthy
4. ✅ Starts backend API
5. ✅ Creates database tables automatically

### Step 3: Access Services

- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **MySQL**: localhost:3306

---

## Detailed Steps

### 1. Prepare Environment File

**Edit `.env.docker.local`:**

```env
# Google Gemini API Key (REQUIRED)
# Get from: https://makersuite.google.com/app/apikey
GOOGLE_API_KEY=AIzaSyXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

# Email Configuration (REQUIRED for alerts)
EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=587
EMAIL_USER=youremail@gmail.com
EMAIL_PASSWORD=your_16_char_app_password
EMAIL_FROM=youremail@gmail.com
```

### 2. Build Docker Image

```bash
# Build without starting
docker-compose build

# Or build with no cache (clean build)
docker-compose build --no-cache
```

**Build time:**
- First time: ~5-10 minutes
- Subsequent: ~1-2 minutes (cached layers)

**Output:**
```
[+] Building 245.3s (18/18) FINISHED
 => [builder 1/5] FROM python:3.11-slim
 => [builder 2/5] WORKDIR /app
 => [builder 3/5] COPY app/requirements.txt
 => [builder 4/5] RUN pip install...
 => [stage-1 1/4] FROM python:3.11-slim
 => [stage-1 2/4] RUN apt-get update...
 => [stage-1 3/4] COPY --from=builder...
 => [stage-1 4/4] COPY app/ /app/
 => exporting to image
 => => naming to docker.io/library/cctv-backend
```

### 3. Start Services

**Option A: Foreground (see logs in real-time)**
```bash
docker-compose --env-file .env.docker.local up
```

**Option B: Background (detached mode)**
```bash
docker-compose --env-file .env.docker.local up -d
```

**First startup output:**
```
Creating network "cctv_default" with the default driver
Creating volume "cctv_mysql_data" with local driver
Creating luggage_monitoring_db ... done
Waiting for database health check...
Creating luggage_monitoring_backend ... done
```

### 4. Verify Services Are Running

```bash
# Check running containers
docker-compose ps

# Expected output:
NAME                         STATUS                   PORTS
luggage_monitoring_backend   Up (healthy)            0.0.0.0:8000->8000/tcp
luggage_monitoring_db        Up (healthy)            0.0.0.0:3306->3306/tcp
```

### 5. Check Logs

```bash
# View all logs
docker-compose logs

# Follow logs in real-time
docker-compose logs -f

# View specific service logs
docker-compose logs backend
docker-compose logs db

# Follow backend logs
docker-compose logs -f backend
```

**Healthy backend logs:**
```
luggage_monitoring_backend | INFO:     Started server process [1]
luggage_monitoring_backend | INFO:     Waiting for application startup.
luggage_monitoring_backend | INFO:     Application startup complete.
luggage_monitoring_backend | INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 6. Test the API

**Health Check:**
```bash
curl http://localhost:8000/health
```

**Expected response:**
```json
{
  "status": "healthy",
  "version": "2.0.0"
}
```

**API Documentation:**
- Open browser: http://localhost:8000/docs
- Try the interactive API

---

## Testing with Frontend

### Update Frontend Configuration

**Frontend/.env:**
```env
VITE_API_BASE_URL=http://localhost:8000
```

**Start frontend:**
```bash
cd Frontend
npm run dev
```

**Access:**
- Frontend: http://localhost:5173
- Backend: http://localhost:8000

---

## Common Commands

### Start/Stop Services

```bash
# Start services (if already built)
docker-compose --env-file .env.docker.local up -d

# Stop services (keeps containers)
docker-compose stop

# Stop and remove containers
docker-compose down

# Stop and remove containers + volumes (DELETES DATABASE!)
docker-compose down -v
```

### View Status

```bash
# Check status
docker-compose ps

# View resource usage
docker stats
```

### Access Container Shell

```bash
# Backend shell
docker-compose exec backend bash

# Inside container:
python -c "from models.database import Base, engine; print('DB connection OK')"
exit

# MySQL shell
docker-compose exec db mysql -u luggage_user -pluggage_pass luggage_monitoring
```

### Restart Services

```bash
# Restart all services
docker-compose restart

# Restart specific service
docker-compose restart backend
docker-compose restart db
```

---

## Database Management

### Initialize Database Tables

Tables are created automatically on first startup. To manually initialize:

```bash
# Access backend container
docker-compose exec backend bash

# Create tables
python -c "from models.database import Base, engine; Base.metadata.create_all(bind=engine); print('Tables created!')"

exit
```

### Access MySQL Database

```bash
# Via docker-compose
docker-compose exec db mysql -u luggage_user -pluggage_pass luggage_monitoring

# Or via localhost
mysql -h 127.0.0.1 -P 3306 -u luggage_user -pluggage_pass luggage_monitoring
```

**MySQL Commands:**
```sql
-- Show tables
SHOW TABLES;

-- View videos
SELECT id, filename, status FROM videos;

-- Count detections
SELECT COUNT(*) FROM detections;

-- View summaries
SELECT * FROM summaries;

-- Exit
exit;
```

### Backup Database

```bash
# Backup to file
docker-compose exec db mysqldump -u luggage_user -pluggage_pass luggage_monitoring > backup.sql

# Restore from backup
docker-compose exec -T db mysql -u luggage_user -pluggage_pass luggage_monitoring < backup.sql
```

---

## Development Workflow

### Code Changes (Live Reload)

Since `app/` is mounted as a volume, you can edit code and see changes:

1. Edit Python files in `app/`
2. Restart backend container:
   ```bash
   docker-compose restart backend
   ```
3. Changes are applied immediately

### Rebuild After Dependency Changes

If you modify `requirements.txt`:

```bash
# Stop services
docker-compose down

# Rebuild
docker-compose build backend

# Start again
docker-compose --env-file .env.docker.local up -d
```

### View Real-time Logs

```bash
# Follow all logs
docker-compose logs -f

# Follow backend only
docker-compose logs -f backend | grep -i error
```

---

## Troubleshooting

### Issue: Port Already in Use

**Error:**
```
Error starting userland proxy: listen tcp 0.0.0.0:8000: bind: address already in use
```

**Solution:**

**Find process using port:**
```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# macOS/Linux
lsof -ti:8000 | xargs kill -9

# Or change port in docker-compose.yml
ports:
  - "8001:8000"  # Use port 8001 instead
```

### Issue: MySQL Connection Failed

**Error:**
```
sqlalchemy.exc.OperationalError: (pymysql.err.OperationalError) (2003, "Can't connect to MySQL server")
```

**Solution:**

1. **Check MySQL is healthy:**
   ```bash
   docker-compose ps
   ```
   Should show `Up (healthy)` for db

2. **Check MySQL logs:**
   ```bash
   docker-compose logs db
   ```

3. **Restart database:**
   ```bash
   docker-compose restart db
   ```

4. **Wait for health check** (can take 30 seconds)

### Issue: Out of Memory

**Error:**
```
ERROR: for backend  Container exceeded memory limit
```

**Solution:**

1. **Increase Docker memory:**
   - Docker Desktop → Settings → Resources
   - Increase Memory to at least 4GB

2. **Use smaller model:**
   ```bash
   # Add to docker-compose.yml under backend environment:
   - YOLO_MODEL=yolov8n.pt
   - FRAME_SKIP=40
   ```

### Issue: Build Fails

**Error:**
```
ERROR [builder 4/5] RUN pip install...
```

**Solution:**

1. **Check internet connection**

2. **Clear Docker cache:**
   ```bash
   docker-compose build --no-cache
   ```

3. **Check requirements.txt** for syntax errors

### Issue: Permission Denied (Linux)

**Error:**
```
mkdir: cannot create directory '/app/uploads': Permission denied
```

**Solution:**

```bash
# Create directories locally first
mkdir -p uploads processed temp

# Set permissions
chmod 777 uploads processed temp

# Or run with sudo
sudo docker-compose up
```

### Issue: YOLO Model Download Fails

**Error:**
```
Failed to download yolov8n.pt
```

**Solution:**

1. **Download manually:**
   ```bash
   wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt
   ```

2. **Copy to container:**
   ```bash
   docker cp yolov8n.pt luggage_monitoring_backend:/app/
   ```

3. **Restart:**
   ```bash
   docker-compose restart backend
   ```

---

## Cleanup

### Remove Containers

```bash
# Stop and remove containers
docker-compose down

# Remove containers + volumes (DELETES DATABASE!)
docker-compose down -v

# Remove containers + volumes + images
docker-compose down -v --rmi all
```

### Free Up Disk Space

```bash
# Remove unused images
docker image prune

# Remove unused volumes
docker volume prune

# Remove everything unused
docker system prune -a
```

---

## Configuration Details

### MySQL Database

**Credentials** (in docker-compose.yml):
- Host: `db` (from backend) or `localhost` (from host)
- Port: `3306`
- User: `luggage_user`
- Password: `luggage_pass`
- Database: `luggage_monitoring`

**Volume:**
- Data persists in `mysql_data` volume
- Located at: `~/.local/share/docker/volumes/cctv_mysql_data`

### Backend API

**Environment:**
- `ENVIRONMENT=production`
- Uses MySQL (not SQLite)
- Mounts: `uploads/`, `processed/`, `temp/`

**Volumes:**
- `./uploads:/app/uploads` - Uploaded videos
- `./processed:/app/processed` - Processed frames
- `./temp:/app/temp` - Temporary files

---

## Performance Tips

### Optimize Build Time

```bash
# Use BuildKit for faster builds
DOCKER_BUILDKIT=1 docker-compose build
```

### Limit Resource Usage

**docker-compose.yml:**
```yaml
services:
  backend:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
        reservations:
          memory: 1G
```

### Use Specific Image Tags

```yaml
services:
  db:
    image: mysql:8.0.33  # Specific version
```

---

## Production Deployment

For production with Docker:

1. **Use Docker Swarm or Kubernetes**
2. **Set up reverse proxy** (nginx)
3. **Use secrets management**
4. **Set up monitoring**
5. **Configure backups**

**Simple production docker-compose:**
```yaml
services:
  backend:
    environment:
      - ENVIRONMENT=production
      - SECRET_KEY=${SECRET_KEY}
    restart: always
    deploy:
      replicas: 2
```

---

## Next Steps

Once Docker deployment is working:

1. ✅ Test video upload
2. ✅ Verify detection works
3. ✅ Test email alerts
4. ✅ Try LLM queries
5. ✅ Check database records
6. ✅ Monitor logs for errors

---

## Resources

- **Docker Docs**: https://docs.docker.com/
- **Docker Compose**: https://docs.docker.com/compose/
- **MySQL Docker**: https://hub.docker.com/_/mysql
- **Python Docker**: https://hub.docker.com/_/python

---

**Your system is now running in Docker! 🐳**

Access the API at: http://localhost:8000/docs
