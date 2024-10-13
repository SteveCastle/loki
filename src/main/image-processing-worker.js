const fs = require('fs');
const os = require('os');
const sharp = require('sharp');
const path = require('path');
const { promisify } = require('util');
const execFile = promisify(require('child_process').execFile);
const crypto = require('crypto');
const workerpool = require('workerpool');

const isDev = process.env.NODE_ENV === 'development';
const isMac = os.platform() === 'darwin';
const ffmpegPath = isDev
  ? path.join(__dirname, 'resources/bin/ffmpeg')
  : path.join(__dirname, '../../../bin/ffmpeg');
const ffProbePath = isDev
  ? path.join(__dirname, 'resources/bin/ffprobe')
  : path.join(__dirname, '../../../bin/ffprobe');

function createHash(input) {
  return crypto.createHash('sha256').update(input).digest('hex');
}

const FileTypes = {
  Image: 'image',
  Video: 'video',
  Audio: 'audio',
  Document: 'document',
  Other: 'other',
};

const Extensions = {
  Image: /\.(jpg|jpeg|png|bmp|svg|jfif|pjpeg|pjp|webp)$/i,
  Video: /\.(mp4|webm|ogg|mkv|mov|gif)$/i,
  Audio: /\.(mp3|wav)$/i,
  Document: /\.(pdf|doc|docx|xls|xlsx|ppt|pptx|txt|csv)$/i,
};

const getFileType = (fileName) => {
  const extension = path.extname(fileName).toLowerCase();
  if (Extensions.Image.test(extension)) return FileTypes.Image;
  if (Extensions.Video.test(extension)) return FileTypes.Video;
  if (Extensions.Audio.test(extension)) return FileTypes.Audio;
  if (Extensions.Document.test(extension)) return FileTypes.Document;
  return FileTypes.Other;
};

const cacheSizes = {
  thumbnail_path_1200: 1200,
  thumbnail_path_600: 600,
  thumbnail_path_100: 100,
};

async function createThumbnail(filePath, basePath, cache, timeStamp) {
  const thumbnailBasePath = path.join(basePath, cache);
  await fs.promises.mkdir(thumbnailBasePath, { recursive: true });

  const thumbnailFileName = createHash(
    filePath + (timeStamp > 0 ? timeStamp.toString() : '')
  );
  let thumbnailFullPath = path.join(thumbnailBasePath, thumbnailFileName);

  const fileType = getFileType(filePath);
  if (fileType === FileTypes.Video) {
    thumbnailFullPath += '.mp4';
    await createVideoThumbnail(filePath, thumbnailFullPath, timeStamp);
  } else if (fileType === FileTypes.Image) {
    await createImageThumbnail(filePath, thumbnailFullPath, cache);
  } else {
    throw new Error('Unsupported file type');
  }

  return thumbnailFullPath;
}

async function createImageThumbnail(filePath, thumbnailFullPath, cache) {
  const imageBuffer = await fs.promises.readFile(filePath);
  await sharp(imageBuffer)
    .resize(cacheSizes[cache])
    .webp()
    .toFile(thumbnailFullPath);
}

async function getVideoMetadata(videoFilePath) {
  try {
    const { stdout } = await execFile(ffProbePath, [
      '-v',
      'quiet',
      '-print_format',
      'json',
      '-show_format',
      '-show_streams',
      videoFilePath,
    ]);
    return JSON.parse(stdout);
  } catch (error) {
    console.error('Error getting video metadata:', error);
    throw error;
  }
}

async function generateVideoThumbnail(
  videoFilePath,
  thumbnailTime,
  thumbnailFullPath,
  useMiddle
) {
  const ffmpegArgs = [
    '-y',
    useMiddle ? '-ss' : '-i',
    useMiddle ? thumbnailTime : videoFilePath,
    useMiddle ? '-i' : '-ss',
    useMiddle ? videoFilePath : thumbnailTime,
    '-vf',
    "scale='min(400,iw)':'min(400,ih)':force_original_aspect_ratio=decrease,pad=ceil(iw/2)*2:ceil(ih/2)*2",
    '-t',
    '2',
    '-an',
    thumbnailFullPath,
  ];

  try {
    await execFile(ffmpegPath, ffmpegArgs);
  } catch (error) {
    console.error('Error generating video thumbnail:', error);
    throw error;
  }
}

async function createVideoThumbnail(
  videoFilePath,
  thumbnailFullPath,
  timeStamp
) {
  try {
    const metadata = await getVideoMetadata(videoFilePath);
    const duration_sec = metadata.format.duration || 0;
    const thumbnailTime = timeStamp || duration_sec / 2;
    const useMiddle = duration_sec > 6;

    await generateVideoThumbnail(
      videoFilePath,
      thumbnailTime,
      thumbnailFullPath,
      useMiddle
    );
  } catch (err) {
    console.error('Error during thumbnail generation', err);
    throw err;
  }
}

workerpool.worker({
  createThumbnail: createThumbnail,
});
