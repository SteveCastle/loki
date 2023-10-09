import fs from 'fs';
import path from 'path';
import { exec } from 'child_process';
const isDev = require('electron-is-dev');
import { promisify } from 'util';

const exifToolPath = isDev
  ? path.join(__dirname, 'resources/bin/exiftool')
  : path.join(__dirname, '../../../bin/exiftool');

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
  stableDiffusionMetaData?: StableDiffusionMetaData;
  extendedMetadata?: ExtendedMetadata;
}

export interface StableDiffusionMetaData {
  prompt: string;
  negativePrompt: string;
  model: string;
}

export interface FileMetadata {
  size: string;
  modified: Date;
  width?: number;
  height?: number;
}

// Interface with tags where tags is an object with string keys and an array of string values
interface ExtendedMetadata {
  tags: { [key: string]: string[] };
}

type FileMetadataInput = [string];
type ExifData = {
  ImageHeight: number;
  ImageWidth: number;
  Parameters: string;
};

type ArtObject = {
  prompt: string;
  negativePrompt: string;
  steps: number;
  sampler: string;
  cfgScale: number;
  seed: number;
  size: string;
  modelHash: string;
  model: string;
};

function parseStableDiffusionMetaData(inputString: string): ArtObject {
  // Split the string at 'Negative prompt:' to separate the prompt and the rest
  const [prompt, rest] = inputString.split('Negative prompt: ');
  // Split the rest at 'Steps:' to separate the negative prompt and the rest
  const [negativePrompt, rest2] = rest.split('Steps: ');
  // Split the rest at 'Sampler:' to separate the steps and the rest
  const [steps, rest3] = rest2.split('Sampler: ');
  // Split the rest at 'cfg_scale:' to separate the sampler and the rest
  const [sampler, rest4] = rest3.split('CFG scale: ');
  // Split the rest at 'Seed:' to separate the cfg_scale and the rest
  const [cfgScale, rest5] = rest4.split('Seed: ');
  // Split the rest at 'Size:' to separate the seed and the rest
  const [seed, rest6] = rest5.split('Size: ');
  // Split the rest at 'Model hash:' to separate the size and the rest
  const [size, rest7] = rest6.split('Model hash: ');
  // Split the rest at 'Model:' to separate the model hash and the rest
  const [modelHash, model] = rest7.split('Model: ');

  return {
    prompt,
    negativePrompt,
    steps: parseInt(steps),
    sampler,
    cfgScale: parseFloat(cfgScale),
    seed: parseInt(seed),
    size,
    modelHash,
    model,
  };
}

async function getExif(filePath: string): Promise<ExifData | null> {
  try {
    // -j flag to output JSON format
    const { stdout, stderr } = await execPromisified(
      `${exifToolPath} -j "${filePath}"`
    );

    if (stderr) {
      console.error(`Error: ${stderr}`);
      return null;
    }

    const metadata = JSON.parse(stdout);
    return metadata[0];
  } catch (error) {
    console.error(`Failed to fetch metadata: ${error}`);
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

async function loadFileMetaData(
  _: Event,
  args: FileMetadataInput
): Promise<Metadata> {
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

  let exif;
  let parameters;
  try {
    exif = await getExif(absolutePath);
    parameters = exif ? exif['Parameters'] : null;
  } catch (e) {
    exif = null;
    parameters = null;
  }
  return {
    fileMetadata: {
      size: formatFileSize(stats.size),
      modified: stats.mtime,
      height: exif ? exif['ImageHeight'] : 0,
      width: exif ? exif['ImageWidth'] : 0,
    },
    extendedMetadata,
    stableDiffusionMetaData: parameters
      ? parseStableDiffusionMetaData(parameters)
      : undefined,
  };
}

export { loadFileMetaData };
