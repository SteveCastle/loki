import fs from 'fs';
import path from 'path';
const { exec } = require('node:child_process');
import parseVttFile, { VttCue } from './parse-vtt';

async function isPathToFile(filePath: string): Promise<boolean> {
  try {
    const stats = await fs.promises.stat(filePath);
    return stats.isFile();
  } catch (err: any) {
    if (err.code === 'ENOENT') {
      return false;
    }
    throw err;
  }
}
async function loadTranscript(filePath: string) {
  let actualFilePath;
  // The transcript path is the same as the media path but with the vtt extension instead of the media extension.
  const pathOptions = [
    filePath.replace(/\.[^/.]+$/, '.vtt'),
    filePath + '.vtt',
  ];
  for (const pathOption of pathOptions) {
    if (await isPathToFile(pathOption)) {
      actualFilePath = pathOption;
      break;
    }
  }
  if (!actualFilePath) {
    return null;
  }

  return parseVttFile(actualFilePath);
}

async function _generateTranscript(mediaPath: string) {
  // Call the command line command whisper to generate the transcript.
  await new Promise((resolve, reject) => {
    const outputDir = path.dirname(mediaPath);
    const child = exec(
      `whisper --output_format vtt --output_dir "${outputDir}" "${mediaPath}"`,
      (error: any, stdout: any, stderr: any) => {
        if (error) {
          console.log(`error: ${error.message}`);
          reject(error);
        }
        if (stderr) {
          console.log(`stderr: ${stderr}`);
          reject(stderr);
        }
        console.log(`stdout: ${stdout}`);
        resolve(stdout);
      }
    );
  });
}

type GenerateTranscriptInput = string;
const generateTranscript = async (mediaPath: GenerateTranscriptInput) => {
  const transcript = await _generateTranscript(mediaPath);
  return transcript;
};

const checkIfWhisperIsInstalled = async (): Promise<boolean> => {
  try {
    await new Promise((resolve, reject) => {
      exec(`whisper --version`, (error: any, stdout: any) => {
        if (error) {
          console.log(`error: ${error.message}`);
          reject(error);
        }
        resolve(stdout);
      });
    });
  } catch (err) {
    console.log('Whisper is not installed.');
    console.log('Please install whisper to use this feature.');
    return false;
  }
  return true;
};

function writeVttFile(filePath: string, cues: VttCue[]): Promise<void> {
  return new Promise((resolve, reject) => {
    const vttContent = ['WEBVTT', ''];
    
    cues.forEach((cue, index) => {
      if (index > 0) {
        vttContent.push('');
      }
      vttContent.push(`${cue.startTime} --> ${cue.endTime}`);
      vttContent.push(cue.text);
    });
    
    vttContent.push('');
    
    fs.writeFile(filePath, vttContent.join('\n'), 'utf8', (err) => {
      if (err) {
        reject(err);
      } else {
        resolve();
      }
    });
  });
}

type ModifyTranscriptInput = {
  mediaPath: string;
  cueIndex: number;
  startTime?: string;
  endTime?: string;
  text?: string;
};

async function findTranscriptPath(mediaPath: string): Promise<string | null> {
  const pathOptions = [
    mediaPath.replace(/\.[^/.]+$/, '.vtt'),
    mediaPath + '.vtt',
  ];
  for (const pathOption of pathOptions) {
    if (await isPathToFile(pathOption)) {
      return pathOption;
    }
  }
  return null;
}

async function modifyTranscript(input: ModifyTranscriptInput): Promise<boolean> {
  const { mediaPath, cueIndex, startTime, endTime, text } = input;

  const transcriptPath = await findTranscriptPath(mediaPath);
  if (!transcriptPath) {
    throw new Error('Transcript file not found');
  }

  const cues = await parseVttFile(transcriptPath);
  if (cueIndex < 0 || cueIndex >= cues.length) {
    throw new Error('Invalid cue index');
  }

  if (startTime !== undefined) cues[cueIndex].startTime = startTime;
  if (endTime !== undefined) cues[cueIndex].endTime = endTime;
  if (text !== undefined) cues[cueIndex].text = text;

  await writeVttFile(transcriptPath, cues);
  return true;
}

type DeleteTranscriptCueInput = {
  mediaPath: string;
  cueIndex: number;
};

async function deleteTranscriptCue(input: DeleteTranscriptCueInput): Promise<boolean> {
  const { mediaPath, cueIndex } = input;
  const transcriptPath = await findTranscriptPath(mediaPath);
  if (!transcriptPath) {
    throw new Error('Transcript file not found');
  }
  const cues = await parseVttFile(transcriptPath);
  if (cueIndex < 0 || cueIndex >= cues.length) {
    throw new Error('Invalid cue index');
  }
  cues.splice(cueIndex, 1);
  await writeVttFile(transcriptPath, cues);
  return true;
}

type InsertTranscriptCueInput = {
  mediaPath: string;
  startTime: string;
  endTime: string;
  text?: string;
};

/**
 * Inserts a new cue and re-sorts by start time so the file stays in
 * canonical order. Returns the index where the new cue ended up so the
 * caller can focus it in the UI.
 */
async function insertTranscriptCue(
  input: InsertTranscriptCueInput
): Promise<number> {
  const { mediaPath, startTime, endTime } = input;
  const text = input.text ?? '';
  const transcriptPath = await findTranscriptPath(mediaPath);
  if (!transcriptPath) {
    throw new Error('Transcript file not found');
  }
  const cues = await parseVttFile(transcriptPath);
  const newCue: VttCue = { startTime, endTime, text };
  cues.push(newCue);
  cues.sort((a, b) => (a.startTime < b.startTime ? -1 : a.startTime > b.startTime ? 1 : 0));
  await writeVttFile(transcriptPath, cues);
  return cues.indexOf(newCue);
}

export {
  loadTranscript,
  generateTranscript,
  checkIfWhisperIsInstalled,
  modifyTranscript,
  deleteTranscriptCue,
  insertTranscriptCue,
};
