# Deployment Guide - Cephalometric Landmark Detection

## Current Deployment Status

✅ **Your application is now live on the internet!**

### Public URL
- **Frontend**: https://207d-113-19-181-190.ngrok-free.app

### What's Running
- ✅ Backend API server on port 8000
- ✅ Frontend dev server on port 5173
- ✅ ngrok tunnel forwarding port 5173 to the internet

## How It Works

1. **ngrok tunnel** creates a secure public URL that forwards to your local server
2. **Frontend** is served from the ngrok URL
3. **API calls** go to the same domain (relative URLs), so they work through the tunnel

## ngrok Setup and Commands

### Prerequisites
1. **Download ngrok**: https://ngrok.com/download
2. **Sign up for free account**: https://dashboard.ngrok.com/signup
3. **Get your authtoken**: https://dashboard.ngrok.com/get-started/your-authtoken

### Initial Setup (One-time)

1. **Install ngrok authtoken**:
   ```bash
   ngrok config add-authtoken YOUR_AUTH_TOKEN_HERE
   ```

### Starting the Application

#### Step 1: Start the Backend Server
```bash
# In the project root directory
python api_server.py
```
- Server runs on http://localhost:8000
- Keep this terminal open

#### Step 2: Start the Frontend Dev Server
```bash
# In a new terminal
cd web
npm run dev
```
- Frontend runs on http://localhost:5173
- Keep this terminal open

#### Step 3: Start ngrok Tunnel
```bash
# In a third terminal
ngrok http 5173
```

Expected output:
```
ngrok                                                                                                                         
                                                                                                                              
Session Status                online                                                        
Account                       Your Name (Plan: Free)                                        
Version                       3.37.3                                                        
Region                        United States (us-cal-1)                                      
Web Interface                 http://127.0.0.1:4040                                        
Forwarding                    https://abcd-1234-5678.ngrok-free.app -> http://localhost:5173

Connections                   ttl     opn     rt1     rt5     p50     p90
                              0       0       0.00    0.00    0.00    0.00
```

**Your public URL is**: `https://abcd-1234-5678.ngrok-free.app`

### Managing ngrok

#### Check ngrok Status
```bash
# Check if ngrok is running
tasklist | findstr ngrok

# Get current tunnel URL
curl http://127.0.0.1:4040/api/tunnels
```

#### Stop ngrok
```bash
# Method 1: Press Ctrl+C in the ngrok terminal
# Method 2: Kill the process
taskkill /F /IM ngrok.exe
```

#### Restart ngrok
```bash
# Stop first, then start again
taskkill /F /IM ngrok.exe
ngrok http 5173
```

### ngrok Web Interface
- **URL**: http://127.0.0.1:4040
- **Features**:
  - View active tunnels
  - Monitor traffic
  - Inspect requests/responses
  - See connection stats

## Common Issues and Solutions

### "ERR_NGROK_8012" Error
**Cause**: ngrok lost connection to local server

**Solution**:
1. Check if frontend is running: `netstat -an | findstr :5173`
2. If not running, restart: `cd web && npm run dev`
3. Restart ngrok: `taskkill /F /IM ngrok.exe && ngrok http 5173`

### "authentication failed" Error
**Cause**: Authtoken not configured

**Solution**:
1. Get authtoken from: https://dashboard.ngrok.com/get-started/your-authtoken
2. Run: `ngrok config add-authtoken YOUR_AUTH_TOKEN`

### Vite Host Error on Mobile
**Cause**: Vite blocks external hosts by default

**Solution**: Already fixed in `vite.config.js`:
```javascript
server: {
  allowedHosts: ['.ngrok-free.app', 'localhost'],
  proxy: {
    '/api': 'http://localhost:8000',
  },
}
```

## Important Notes

### ⚠️ Limitations
- The ngrok URL changes each time you restart ngrok
- Free tier has bandwidth limits (1GB/day)
- 40 connections per minute limit
- Your computer must be running for the app to be accessible
- Single-threaded processing (users queue up)

### ✅ Advantages
- No server costs
- Full functionality (including AI model)
- HTTPS included automatically
- Easy to set up
- Real-time traffic monitoring

## Keeping It Running

### To Stop the Deployment
1. Close the ngrok terminal (Ctrl+C)
2. Stop the frontend (Ctrl+C in its terminal)
3. Stop the backend (Ctrl+C in its terminal)

### To Restart the Deployment
1. Start backend: `python api_server.py`
2. Start frontend: `cd web && npm run dev`
3. Start ngrok: `ngrok http 5173`
4. Note the new URL from ngrok output

### Quick Start Script (Windows)
Create `start_deployment.bat`:
```batch
@echo off
echo Starting Cephalometric App Deployment...
echo.
echo 1. Starting backend...
start "Backend" cmd /k "python api_server.py"
timeout /t 3 /nobreak > nul

echo 2. Starting frontend...
start "Frontend" cmd /k "cd web && npm run dev"
timeout /t 5 /nobreak > nul

echo 3. Starting ngrok...
start "Ngrok" cmd /k "ngrok http 5173"
echo.
echo Deployment started! Check ngrok terminal for public URL.
pause
```

## Making It Permanent (Optional)

For a more permanent deployment, consider:

1. **Cloudflare Tunnel** - Free, stable subdomain
2. **Railway/Render** - Deploy to the cloud (free tier available)
3. **VPS Hosting** - Full control (~$5/month)
4. **Paid ngrok** - Custom domains, higher limits ($8/month)

## Security Considerations

- The app is currently public (no authentication)
- Anyone with the URL can access it
- Consider adding authentication for production use
- Monitor usage via ngrok web interface

## Troubleshooting

### If the app is not accessible:
1. Check all three services are running:
   - Backend: `netstat -an | findstr :8000`
   - Frontend: `netstat -an | findstr :5173`
   - ngrok: `tasklist | findstr ngrok`
2. Check ngrok web interface: http://127.0.0.1:4040
3. Restart services if needed

### If API calls fail:
1. Ensure backend is running on port 8000
2. Check browser console for errors
3. Verify the API endpoint URLs in web/src/api.js

### Performance Issues:
- Monitor ngrok web interface for connection count
- More than 5 concurrent users may experience delays
- Consider cloud deployment for production use
