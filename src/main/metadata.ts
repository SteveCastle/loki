import fs from 'fs';
import path from 'path';
import os from 'os';
import { exec } from 'child_process';
const isDev = require('electron-is-dev');
import { promisify } from 'util';
import { IpcMainInvokeEvent } from 'electron';
import { Database } from './database';

const platform = os.platform();
const isWindows = platform === 'win32';

// Get platform-specific binary path
function getBinaryPath(binaryName: string): string {
  let platformDir;
  let binaryFile = binaryName;
  
  if (platform === 'darwin') {
    platformDir = 'darwin';
  } else if (platform === 'win32') {
    platformDir = 'win32';
    binaryFile = `${binaryName}.exe`;
  } else {
    // Linux
    platformDir = 'linux';
  }
  
  return isDev
    ? path.join(__dirname, 'resources/bin', platformDir, binaryFile)
    : path.join(__dirname, '../../../bin', binaryFile);
}

const ffprobePath = getBinaryPath('ffprobe');

function formatFileSize(size: number): string {
  const i = size === 0 ? 0 : Math.floor(Math.log(size) / Math.log(1024));
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];

  return `${(size / Math.pow(1024, i)).toFixed(2)} ${sizes[i]}`;
}

// Convert fs.stat into a Promise-based function
const stat = promisify(fs.stat);
const execPromisified = promisify(exec);

export interface Metadata {
  fileMetadata: FileMetadata;
  description?: string;
  transcript?: string;
  hash: string;
  extendedMetadata?: ExtendedMetadata;
}

export interface FileMetadata {
  size: string;
  modified: Date;
  width?: number;
  height?: number;
  description?: string;
  transcript?: string;
}

// Interface with tags where tags is an object with string keys and an array of string values
interface ExtendedMetadata {
  tags: { [key: string]: string[] };
}

type FileMetadataInput = [string];

interface MediaDimensions {
  width: number;
  height: number;
}

// Get media dimensions using ffprobe
async function getMediaDimensions(filePath: string): Promise<MediaDimensions | null> {
  try {
    const { stdout } = await execPromisified(
      `"${ffprobePath}" -v error -select_streams v:0 -show_entries stream=width,height -of json "${filePath}"`
    );
    
    const data = JSON.parse(stdout);
    if (data.streams && data.streams[0]) {
      return {
        width: data.streams[0].width || 0,
        height: data.streams[0].height || 0
      };
    }
    return null;
  } catch (error) {
    console.error(`Failed to fetch media dimensions: ${error}`);
    return null;
  }
}

function getExtendedMetadata(jsonPath: string): ExtendedMetadata {
  const data: ExtendedMetadata = { tags: {} };
  const duplicateKeys = new Set();
  try {
    const json = JSON.parse(fs.readFileSync(jsonPath, 'utf8'));
    if (!json.tags && !json.tag_string_general) {
      data.tags = {};
    } else if (
      typeof json.tags === 'string' ||
      typeof json.tag_string_general === 'string'
    ) {
      data.tags['general'] = json.tags
        ? json.tags.split(' ').filter((tag: string) => {
            const isDuplicate = duplicateKeys.has(tag);
            duplicateKeys.add(tag);
            return tag.length > 0 && !isDuplicate;
          })
        : json.tag_string_general.split(' ').filter((tag: string) => {
            const isDuplicate = duplicateKeys.has(tag);
            duplicateKeys.add(tag);
            return tag.length > 0 && !isDuplicate;
          });
    } else if (Array.isArray(json.tags)) {
      data.tags['general'] = json.tags.filter((tag: string) => {
        const isDuplicate = duplicateKeys.has(tag);
        duplicateKeys.add(tag);
        return tag.length > 0 && !isDuplicate;
      });
    } else if (typeof json.tags === 'object') {
      data.tags = json.tags;
    }
  } catch (e) {
    // ignore
  }
  return data;
}

const loadFileMetaData =
  (db: Database) =>
  async (_: IpcMainInvokeEvent, args: FileMetadataInput): Promise<Metadata> => {
    const [filePath] = args;
    const absolutePath = path.resolve(filePath);
    let stats;
    try {
      stats = await stat(absolutePath);
    } catch (e) {
      return {
        fileMetadata: {
          size: '0 Bytes',
          modified: new Date(),
          height: 0,
          width: 0,
        },
      };
    }
    // load json from same path as file with .json appended
    const jsonPath = absolutePath + '.json';
    const extendedMetadata: ExtendedMetadata = getExtendedMetadata(jsonPath);

    const media = await db.get('SELECT * FROM media WHERE path = ?', [
      absolutePath,
    ]);

    // Get media dimensions using ffprobe
    let dimensions;
    try {
      dimensions = await getMediaDimensions(absolutePath);
    } catch (e) {
      dimensions = null;
    }
    
    return {
      description: media?.description,
      transcript: media?.transcript,
      hash: media?.hash,
      fileMetadata: {
        size: formatFileSize(stats.size),
        modified: stats.mtime,
        height: dimensions ? dimensions.height : 0,
        width: dimensions ? dimensions.width : 0,
      },
      extendedMetadata,
    };
  };

export interface GifMetadata {
  frameCount: number;
  duration: number; // in milliseconds
}

// Get GIF metadata using ffprobe (frame count and total duration)
async function getGifMetadata(filePath: string): Promise<GifMetadata | null> {
  try {
    // Get frame count
    const { stdout: frameCountOutput } = await execPromisified(
      `"${ffprobePath}" -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of json "${filePath}"`
    );

    // Get duration
    const { stdout: durationOutput } = await execPromisified(
      `"${ffprobePath}" -v error -show_entries format=duration -of json "${filePath}"`
    );

    const frameData = JSON.parse(frameCountOutput);
    const durationData = JSON.parse(durationOutput);

    const frameCount = frameData.streams?.[0]?.nb_read_frames
      ? parseInt(frameData.streams[0].nb_read_frames, 10)
      : 1;

    // Duration is in seconds, convert to milliseconds
    const durationSeconds = durationData.format?.duration
      ? parseFloat(durationData.format.duration)
      : 0;
    const duration = Math.round(durationSeconds * 1000);

    return { frameCount, duration };
  } catch (error) {
    console.error(`Failed to fetch GIF metadata: ${error}`);
    return null;
  }
}

type GetGifMetadataInput = [string];
const loadGifMetadata =
  () =>
  async (_: IpcMainInvokeEvent, args: GetGifMetadataInput): Promise<GifMetadata | null> => {
    const [filePath] = args;
    return getGifMetadata(filePath);
  };

export { loadFileMetaData, loadGifMetadata };
