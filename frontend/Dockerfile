FROM node:20-alpine AS build

WORKDIR /app
COPY package.json ./
RUN npm install
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/nginx.conf
# Remove the default conf.d include so the entrypoint script cannot
# regenerate a port-80 server block that would shadow our config
RUN rm -f /etc/nginx/conf.d/default.conf
EXPOSE 5173
