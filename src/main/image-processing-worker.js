const fs = require('fs');
const os = require('os');
const sharp = require('sharp');
const path = require('path');
const { exec } = require('child_process');
const crypto = require('crypto');
const workerpool = require('workerpool');

const isDev = process.env.NODE_ENV === 'development';
const isMac = os.platform() === 'darwin';
const ffmpegPath = isMac
  ? 'ffmpeg'
  : isDev
  ? path.join(__dirname, 'resources/bin/ffmpeg')
  : path.join(__dirname, '../../../bin/ffmpeg');
const ffProbePath = isMac
  ? 'ffprobe'
  : isDev
  ? path.join(__dirname, 'resources/bin/ffprobe')
  : path.join(__dirname, '../../../bin/ffprobe');

function createHash(input) {
  const hash = crypto.createHash('sha256');
  hash.update(input);
  return hash.digest('hex');
}

const FileTypes = {
  Image: 'image',
  Video: 'video',
  Audio: 'audio',
  Document: 'document',
  Other: 'other',
};

const Extensions = {
  Image: 'jpg|jpeg|png|bmp|svg|jfif|pjpeg|pjp|webp',
  Video: 'mp4|webm|ogg|mkv|gif',
  Audio: 'mp3|wav',
  Document: 'pdf|doc|docx|xls|xlsx|ppt|pptx|txt|csv',
};

const getFileType = (fileName) => {
  const extension = fileName.split('.').pop()?.toLowerCase();
  if (extension) {
    if (Extensions.Image.includes(extension)) {
      return FileTypes.Image;
    }
    if (Extensions.Video.includes(extension)) {
      return FileTypes.Video;
    }
    if (Extensions.Audio.includes(extension)) {
      return FileTypes.Audio;
    }
    if (Extensions.Document.includes(extension)) {
      return FileTypes.Document;
    }
  }
  return FileTypes.Other;
};

const cacheSizes = {
  thumbnail_path_1200: 1200,
  thumbnail_path_600: 600,
  thumbnail_path_100: 100,
};

// Given a file path, generate a thumbnail, and return the path to the thumbnail.
async function createThumbnail(filePath, basePath, cache, timeStamp) {
  // Parts of the thumbnail path. The filename is a sha256 hash of the input path.
  const thumbnailBasePath = path.join(basePath, cache);

  //Check if thumbnailBasePath exists, if it does not, create it.
  if (!fs.existsSync(thumbnailBasePath)) {
    fs.mkdirSync(thumbnailBasePath, { recursive: true });
  }

  const thumbnailFileName = createHash(
    filePath + (timeStamp > 0 ? timeStamp.toString() : '')
  );
  let thumbnailFullPath = thumbnailBasePath + '/' + thumbnailFileName;
  if (getFileType(filePath) === 'video') {
    thumbnailFullPath = thumbnailFullPath + '.webp';
  }

  const fileType = getFileType(filePath);
  if (fileType === 'video') {
    // Get home directory from node.
    await createVideoThumbnail(filePath, thumbnailFullPath, timeStamp);
    return thumbnailFullPath;
  }
  // Read the image file.
  const imageBuffer = fs.readFileSync(filePath);

  // Create a thumbnail.
  await sharp(imageBuffer)
    .resize(cacheSizes[cache])
    .webp()
    .toFile(thumbnailFullPath);

  // Convert the output Buffer to a Blob.

  return thumbnailFullPath;
}

const getVideoMetadata = (videoFilePath) => {
  return new Promise((resolve, reject) => {
    exec(
      `${ffProbePath} -v quiet -print_format json -show_format -show_streams "${videoFilePath}"`,
      (err, stdout, stderr) => {
        if (err) {
          reject(err);
        } else {
          resolve(JSON.parse(stdout));
        }
      }
    );
  });
};

const generateUpscaledThumbnail = (imageFilePath, thumbnailFullPath) => {
  return new Promise((resolve, reject) => {
    exec(`up -i ${imageFilePath} -o ${thumbnailFullPath}.webp -s 2`)
      .on('exit', (code) => {
        if (code !== 0) {
          return reject(new Error(`up exited with code ${code}`));
        }
        resolve();
      })
      .on('error', reject);
  });
};

// Calls ffmpeg thumbnail function to generate a thumbnail.
const generateVideoThumbnail = (
  videoFilePath,
  thumbnailTime,
  thumbnailFullPath,
  useMiddle
) => {
  return new Promise((resolve, reject) => {
    if (useMiddle) {
      exec(
        `${ffmpegPath} -y -ss ${thumbnailTime} -i "${videoFilePath}" -vf "scale=600:-1" -loop 0 -t 2 -an ${thumbnailFullPath}
      `
      )
        .on('exit', (code) => {
          if (code !== 0) {
            return reject(new Error(`ffmpeg exited with code ${code}`));
          }
          resolve();
        })
        .on('error', (err) => {
          reject(err);
        });
    } else {
      exec(
        `${ffmpegPath} -y -i "${videoFilePath}" -vf "scale=600:-1" -loop 0 -an ${thumbnailFullPath}
        `
      )
        .on('exit', (code) => {
          if (code !== 0) {
            return reject(new Error(`ffmpeg exited with code ${code}`));
          }
          resolve();
        })
        .on('error', reject);
    }
  });
};

const createVideoThumbnail = async (
  videoFilePath,
  thumbnailFullPath,
  timeStamp
) => {
  try {
    const metadata = await getVideoMetadata(videoFilePath);
    const duration_sec = metadata.format.duration || 0;
    const thumbnailTime = timeStamp ? timeStamp : duration_sec / 2;
    const useMiddle = duration_sec > 6;
    console.log('Generating thumbnail in a worker.');
    await generateVideoThumbnail(
      videoFilePath,
      thumbnailTime,
      thumbnailFullPath,
      useMiddle
    );
    console.log('Thumbnail generated in a worker.');
  } catch (err) {
    console.error('Error during thumbnail generation', err);
    return err;
  }
};

workerpool.worker({
  createThumbnail: createThumbnail,
});
