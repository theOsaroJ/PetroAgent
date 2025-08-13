# --- build stage ---
FROM node:18-alpine AS builder
WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci || npm install

COPY tsconfig.json vite.config.ts postcss.config.js tailwind.config.js index.html ./
COPY src ./src
COPY public ./public

RUN npm run build

# --- runtime stage ---
FROM nginx:stable-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=builder /app/dist /usr/share/nginx/html
