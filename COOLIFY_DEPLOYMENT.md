# Coolify Deployment Guide

Deploy the Luggage Monitoring System on Coolify.

## Quick Setup (2 Minutes)

### Step 1: Connect Your Repository

1. In Coolify, create a new **Docker Compose** project
2. Connect your Git repository (GitHub/GitLab)
3. Select branch: `main` (or your default branch)

### Step 2: Configure Build Settings

Set these in Coolify:

- **Base Directory**: `app`
- **Dockerfile Location**: `Dockerfile.simple`
- **Docker Compose File**: `docker-compose.simple.yml`

Or if you want to use the optimized build (might fail on some networks):
- **Dockerfile Location**: `Dockerfile`
- **Docker Compose File**: `docker-compose.yml`

### Step 3: Set Environment Variables

Add these environment variables in Coolify:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_gmail_app_password
FROM_EMAIL=your_email@gmail.com
ALERT_RECIPIENTS=recipient@example.com
CORS_ORIGINS=http://localhost:3000,http://localhost:5173,https://yourdomain.com
```

**Important**:
- Get Google Gemini API key from: https://makersuite.google.com/app/apikey
- Gmail App Password: https://myaccount.google.com/apppasswords
- CORS_ORIGINS: Add your frontend domain (NO SPACES!)

### Step 4: Deploy

Click **Deploy** in Coolify. The build will take 5-10 minutes.

---

## Troubleshooting

### Build Fails with apt-get errors

**Solution**: Use `Dockerfile.simple` instead of `Dockerfile`

In Coolify:
- Change **Dockerfile Location** to: `Dockerfile.simple`
- Change **Docker Compose File** to: `docker-compose.simple.yml`
- Click **Redeploy**

`Dockerfile.simple` uses the full Python image which has all build tools pre-installed.

### CORS Errors

Make sure `CORS_ORIGINS` in Coolify environment variables includes your frontend URL:

```env
CORS_ORIGINS=https://yourdomain.com,https://www.yourdomain.com
```

**No spaces between URLs!**

### Database Issues

The simple docker-compose uses SQLite (file-based database). The database file will be in the volume mount.

To use MySQL instead, modify `docker-compose.simple.yml` to include the MySQL service from the full `docker-compose.yml`.

### Port Conflicts

Default port is `8000`. If you need a different port, modify in Coolify:
- Container port: `8000`
- Host port: `YOUR_PREFERRED_PORT`

---

## Accessing Your Application

After deployment:

- **API**: `https://your-coolify-domain.com`
- **API Docs**: `https://your-coolify-domain.com/docs`
- **Health Check**: `https://your-coolify-domain.com/health`

---

## Updating the Application

1. Push changes to your Git repository
2. In Coolify, click **Redeploy**
3. Wait for build to complete

Coolify will automatically pull the latest code and rebuild.

---

## Performance Tips

1. **Use smaller YOLO model**: Set in environment
   ```env
   YOLO_MODEL=yolov8n.pt
   ```
   (n = nano, s = small, m = medium, l = large, x = xlarge)

2. **Limit frame processing**:
   ```env
   FRAME_SKIP=30
   ```
   Higher number = faster processing, lower accuracy

3. **Disable face analysis** (if not needed):
   ```env
   ANALYZE_FACES=false
   ```

---

## Support

For Coolify-specific issues:
- Coolify Docs: https://coolify.io/docs
- Coolify Discord: https://coollabs.io/discord

For application issues:
- Check application logs in Coolify dashboard
- Check API documentation at `/docs` endpoint
