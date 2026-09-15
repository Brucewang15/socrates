const GRAFANA_URL = process.env.GRAFANA_URL ?? "http://localhost:3001";

module.exports = {
  // Serve Grafana under /monitoring so dev is one origin. Grafana must also be
  // told it lives under that prefix (GF_SERVER_ROOT_URL and
  // GF_SERVER_SERVE_FROM_SUB_PATH in docker-compose.dev.yml) or it builds asset
  // URLs at / and the page comes back blank.
  async rewrites() {
    return [
      { source: "/monitoring", destination: `${GRAFANA_URL}/monitoring` },
      { source: "/monitoring/:path*", destination: `${GRAFANA_URL}/monitoring/:path*` },
    ];
  },
};
