# reverse-web: Vite build of the TypeScript / three-vrm front end, served by nginx (build context = repository root)
FROM node:22-alpine AS build
WORKDIR /web
COPY reverse_text2sign/service/web/package.json reverse_text2sign/service/web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY reverse_text2sign/service/web/index.html reverse_text2sign/service/web/tsconfig.json reverse_text2sign/service/web/vite.config.ts ./
COPY reverse_text2sign/service/web/src ./src
RUN npm run build

FROM nginx:1.27-alpine
COPY reverse_text2sign/service/web/nginx.conf /etc/nginx/nginx.conf
COPY --from=build /web/dist /usr/share/nginx/html
EXPOSE 80
