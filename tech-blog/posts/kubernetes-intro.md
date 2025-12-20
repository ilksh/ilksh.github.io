---
title: Kubernetes Introduction
category: K8S
date: 2024.03.25
readtime: 15 min
---

# Kubernetes Introduction

Container orchestration with Kubernetes.

## What is Kubernetes?

Kubernetes (K8s) is an open-source container orchestration platform.

## Basic Concepts

- **Pod**: Smallest deployable unit
- **Service**: Network endpoint for pods
- **Deployment**: Manages pod replicas

## Simple Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: my-app
  template:
    metadata:
      labels:
        app: my-app
    spec:
      containers:
      - name: my-app
        image: my-app:latest
        ports:
        - containerPort: 3000
```

## Conclusion

Kubernetes is essential for managing containerized applications at scale.