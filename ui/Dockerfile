# Multi-stage build for React Training UI

# Stage 1: Build
FROM node:20-alpine AS build

WORKDIR /app

# Copy package files
COPY self-learning-agent/ui/package.json self-learning-agent/ui/package-lock.json* ./

# Install dependencies
RUN npm ci --silent

# Copy source code
COPY self-learning-agent/ui/ .

# Build the app
RUN npm run build

# Stage 2: Production
FROM nginx:alpine

# Copy built assets from build stage
COPY --from=build /app/build /usr/share/nginx/html

# Copy custom nginx config
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose port 3001
EXPOSE 3001

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://localhost:3001/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
