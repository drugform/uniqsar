FROM node:12-alpine as builder
WORKDIR /app
COPY frontend/package.json /app/package.json
RUN npm install --only=prod
COPY frontend /app
RUN npm run build


FROM nginx:1.16.0-alpine
COPY --from=builder /app/build /usr/share/nginx/html
COPY frontend/nginx/nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]