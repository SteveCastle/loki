import { randomUUID } from 'crypto';
import { Database } from './database';
import { generateTranscript } from './transcript';
import path from 'path';

export type Job = {
  id: string;
  mediaPaths: string[];
  status: 'pending' | 'started' | 'complete' | 'error';
  error?: string;
  progress: number;
  type: string;
  title: string;
  createdAt: string;
  updatedAt: string;
  invalidations?: string[][];
};

export type JobQueue = Map<string, Job>;
type TaskMap = {
  [key: string]: {
    title: string;
    fn: (...args: any[]) => any;
  };
};
const jobs: JobQueue = new Map<string, Job>();

const tasks: TaskMap = {
  generateTranscript: {
    title: 'Generating Transcript',
    fn: generateTranscript,
  },
};

const asyncJobWorker = async (
  jobId: string,
  invokingWindow: Electron.WebContents
) => {
  const job = jobs.get(jobId);
  if (!job) return;
  const { mediaPaths } = job;
  job.status = 'started';
  jobs.set(jobId, job);
  invokingWindow.send('job-updated', job);
  const total = mediaPaths.length;
  let progress = 0;
  for (let i = 0; i < total; i++) {
    progress = i / total;
    job.progress = progress;
    jobs.set(jobId, job);
    invokingWindow.send('job-updated', job);
    try {
      await tasks[job.type].fn(mediaPaths[i]);
    } catch (err: any) {
      job.status = 'error';
      job.error = `Error: Could not complete ${job.type}`;
      jobs.set(jobId, job);
      invokingWindow.send('job-updated', job);
    }
  }
  job.progress = 1;
  job.status = 'complete';
  jobs.set(jobId, job);
  invokingWindow.send('job-complete', job);
  console.log('completed', job.id);
  //delete job from queue
  jobs.delete(jobId);
  //take next job from queue
  const nextJob = jobs.values().next().value;
  if (!nextJob) return;
  console.log('next job', nextJob.id);
  asyncJobWorker(nextJob.id, invokingWindow);
};

type CreateJobInput = [string[], string, string[][]];
const createJob =
  (db: Database, invokingWindow: Electron.WebContents) =>
  async (_: Event, args: CreateJobInput) => {
    const [mediaPaths, jobType, invalidations] = args;
    const jobId = randomUUID();
    const newJob: Job = {
      id: jobId,
      mediaPaths: mediaPaths.map((mediaPath) => path.normalize(mediaPath)),
      status: 'pending',
      progress: 0,
      type: jobType,
      title: tasks[jobType].title,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      invalidations,
    };
    jobs.set(jobId, newJob);
    console.log('new job', jobId);
    //start job if it is the only job in the queue
    if (jobs.size === 1) {
      asyncJobWorker(jobId, invokingWindow);
    }

    return newJob;
  };

export { createJob };
