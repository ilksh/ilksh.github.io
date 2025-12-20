---
title: Docker Getting Started
category: DOCKER
date: 2024.03.20
readtime: 10 min
---

# Docker Getting Started

A beginner's guide to Docker containerization.

## What is Docker?

Docker is a platform for developing, shipping, and running applications in containers.

## Basic Commands
```bash
# Pull an image
docker pull nginx

# Run a container
docker run -d -p 8080:80 nginx

# List running containers
docker ps

# Stop a container
docker stop [container_id]
```

## Dockerfile Example
```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
EXPOSE 3000
CMD ["npm", "start"]
```

## Docker Compose
```yaml
version: '3.8'
services:
  web:
    build: .
    ports:
      - "3000:3000"
  db:
    image: postgres:14
    environment:
      POSTGRES_PASSWORD: secret
```

> "Containers are the future of software deployment."

## Conclusion

Docker simplifies deployment and ensures consistency across environments.