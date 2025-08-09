# build stage
FROM node:18-alpine AS builder
WORKDIR /app
COPY package.json postcss.config.js tailwind.config.js vite.config.ts index.html ./
COPY public ./public
COPY src ./src
RUN npm ci
RUN npm run build

# serve stage
FROM nginx:stable-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=builder /app/dist /usr/share/nginx/html
EXPOSE 80
