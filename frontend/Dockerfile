# ---------- Build stage ----------
    FROM node:20-alpine AS builder
    
    WORKDIR /app
    
    # Copy dependency files
    COPY package.json package-lock.json* ./
    
    # Install deps
    RUN npm install
    
    # Copy source
    COPY . .
    
    # Build Next.js app
    RUN npm run build
    
    
    # ---------- Runtime stage ----------
    FROM node:20-alpine AS runner
    
    WORKDIR /app
    
    ENV NODE_ENV=production
    
    # Only copy built output + node_modules
    COPY --from=builder /app/.next ./.next
    COPY --from=builder /app/public ./public
    COPY --from=builder /app/node_modules ./node_modules
    COPY --from=builder /app/package.json ./package.json
    COPY --from=builder /app/next.config.ts ./next.config.ts
    
    EXPOSE 3000
    
    CMD ["npm", "start"]
    