# ThaiSLM web (Next.js, standalone output). Build context = repository root (for models/mediapipe/holistic_landmarker.task).
FROM node:22-alpine AS build
WORKDIR /web
COPY webapp/web/package.json webapp/web/package-lock.json* ./
COPY webapp/web/scripts ./scripts
RUN npm ci || npm install
COPY webapp/web/ ./
# the exact MediaPipe Holistic model the server and the training data use
COPY models/mediapipe/holistic_landmarker.task ./public/models/holistic_landmarker.task
RUN node scripts/copy-mediapipe.mjs && npm run build

FROM node:22-alpine
WORKDIR /web
ENV NODE_ENV=production PORT=3000 HOSTNAME=0.0.0.0
COPY --from=build /web/.next/standalone ./
COPY --from=build /web/.next/static ./.next/static
COPY --from=build /web/public ./public
EXPOSE 3000
CMD ["node", "server.js"]
