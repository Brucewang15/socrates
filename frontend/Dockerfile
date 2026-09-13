# Frontend tier. Built from the repo root, not from frontend/:
#
#   docker build -f frontend/Dockerfile -t socrates-frontend .
#   docker run -p 3000:3000 socrates-frontend
#
# NEXT_PUBLIC_* is inlined into the client bundle at build time, not read from
# the environment at runtime -- so the backend URL has to be known here, and a
# deployed image needs rebuilding if that URL changes.
FROM node:22-slim AS build

WORKDIR /app
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
ARG NEXT_PUBLIC_API_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL
RUN npm run build

FROM node:22-slim
WORKDIR /app
ENV NODE_ENV=production
COPY --from=build /app/.next ./.next
COPY --from=build /app/node_modules ./node_modules
COPY --from=build /app/package.json ./package.json
COPY --from=build /app/next.config.js ./next.config.js

EXPOSE 3000
CMD ["npm", "run", "start"]
