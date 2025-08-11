# --- build stage ---
FROM node:18-alpine AS builder
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm install
COPY public ./public
COPY src ./src
COPY index.html ./index.html
COPY tsconfig.json vite.config.ts postcss.config.js tailwind.config.js ./
RUN npm run build

# --- runtime stage ---
FROM nginx:stable-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=builder /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
