# build stage
FROM node:18-alpine AS builder
WORKDIR /app
COPY package.json postcss.config.js tailwind.config.js index.html ./
COPY src ./src
COPY public ./public
RUN npm install
RUN npm run build

# runtime stage
FROM nginx:stable-alpine
WORKDIR /usr/share/nginx/html
COPY --from=builder /app/dist ./
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx","-g","daemon off;"]
