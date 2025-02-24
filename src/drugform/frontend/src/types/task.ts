import { Molecula } from "./molecula";

export interface Task {
  taskId: string;
  userId: string;
  startTime: number;
  endTime?: number | null;
  nIter: number;
  generatorParams: any;
  taskParams: any;
  taskInfo: any;
  name?: string;
}

export interface CreateTaskResponse {
  taskId: string;
}

export interface TaskResult {
  id: number;
  mol?: Molecula;
  value: {
    druglike: { fastprops: { value: number; score: number; report: string } };
    gen_toxicity: {
      toxicity_light: { value: number; score: number; report: string };
    };
    total: { score: number };
  };
  score: number;
}
