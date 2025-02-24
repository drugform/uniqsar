export const endpoints = {
  generate: "/generate",
  generateInfo: "/generate/info",
  encodeCsv: "/api/encodeCSV",
  encodeMol: "/encodeMol",
  calc: "/calc",
  generateInfoByTask: (id: string) => `/generate/${id}/info`,
  generateTaskStop: (id: string) => `/generate/${id}/stop`,
  generateTaskResult: (id: string, count: number) =>
    `/generate/${id}/results/${count}`,
  generateTaskMetrics: (id: string) => `/generate/${id}/metrics`,
};
