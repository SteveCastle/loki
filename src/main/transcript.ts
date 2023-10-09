import fs from 'fs';
import path from 'path';
const { exec } = require('node:child_process');
import parseVttFile from './parse-vtt';

function isPathToFile(path: string) {
  try {
    const stats = fs.statSync(path);
    return stats.isFile();
  } catch (err: any) {
    // If the error is because the file doesn't exist, return false.
    if (err.code === 'ENOENT') {
      return false;
    }
    // For other errors, throw the error to be handled by the caller.
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
  // Test if transcriptPath is an actual file.
  for (const pathOption of pathOptions) {
    if (isPathToFile(pathOption)) {
      actualFilePath = pathOption;
      break;
    }
  }
  if (!actualFilePath) {
    return null;
  }

  const contents = await parseVttFile(actualFilePath);

  // return the metadata
  return contents;
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

export { loadTranscript, generateTranscript, checkIfWhisperIsInstalled };
