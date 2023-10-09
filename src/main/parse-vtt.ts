// Import the necessary modules
import fs from 'fs';
import readline from 'readline';

// Define the interface for a single VTT cue
export interface VttCue {
  startTime: string;
  endTime: string;
  text: string;
}

// Function to parse a VTT file and return a Promise with a JSON object
async function parseVttFile(filepath: string): Promise<VttCue[]> {
  return new Promise((resolve, reject) => {
    const cues: VttCue[] = [];
    let currentCue: VttCue | null = null;

    // Create a readline interface to read the file line by line
    const rl = readline.createInterface({
      input: fs.createReadStream(filepath),
      crlfDelay: Infinity,
    });

    // Process each line
    rl.on('line', (line) => {
      // If the line is empty, it marks the end of the current cue
      if (line.trim() === '' && currentCue !== null) {
        cues.push(currentCue);
        currentCue = null;
      } else if (currentCue === null) {
        // Parse the start and end time of the cue
        const times = line.split(' --> ');
        if (times.length === 2) {
          currentCue = {
            startTime: times[0].trim(),
            endTime: times[1].trim(),
            text: '',
          };
        }
      } else if (
        currentCue !== null &&
        !line.startsWith('NOTE') &&
        !line.startsWith('WEBVTT') &&
        !/^\d+$/.test(line)
      ) {
        // Append the line of text to the current cue
        currentCue.text += (currentCue.text === '' ? '' : '\n') + line;
      }
    });

    // When the readline interface is closed, return the parsed cues
    rl.on('close', () => {
      if (currentCue !== null) {
        cues.push(currentCue);
      }
      resolve(cues);
    });

    // Handle any errors that may occur during the readline process
    rl.on('error', (err) => {
      reject(err);
    });
  });
}

export default parseVttFile;
