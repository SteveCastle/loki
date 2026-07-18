const fs = require('fs');
const path = require('path');
const os = require('os');
const { promisify } = require('util');
const execFile = promisify(require('child_process').execFile);
const crypto = require('crypto');
const workerpool = require('workerpool');
let log;
try {
  log = require('electron-log');
} catch (_) {
  log = console;
}

const isDev = process.env.NODE_ENV === 'development';
const platform = os.platform();
const isWindows = platform === 'win32';

// Get platform-specific binary path
function getBinaryPath(binaryName) {
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

const ffmpegPath = getBinaryPath('ffmpeg');
const ffProbePath = getBinaryPath('ffprobe');

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
  Image: /\.(jpg|jpeg|png|bmp|svg|jfif|pjpeg|pjp|webp|avif)$/i,
  Video: /\.(mp4|webm|ogg|mkv|mov|m4v|gif)$/i,
  Audio: /\.(mp3|wav|flac|aac|m4a|opus|wma|aiff|ape)$/i,
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
  const stat = await fs.promises.stat(filePath);
  if (!stat.isFile() || stat.size === 0) {
    throw new Error('Invalid or empty image file');
  }
  const targetSize = cacheSizes[cache] || 600;
  const scaleExpr = `scale='min(${targetSize},iw)':-2:force_original_aspect_ratio=decrease`;
  // Explicitly set muxer and codec so ffmpeg can write even without a file extension
  const ffmpegArgs = [
    '-y',
    '-i',
    filePath,
    '-vf',
    scaleExpr,
    '-f',
    'image2',
    '-vcodec',
    'png',
    '-frames:v',
    '1',
    thumbnailFullPath,
  ];
  try {
    await execFile(ffmpegPath, ffmpegArgs);
  } catch (error) {
    try {
      log.error('FFmpeg image thumbnail failed:', filePath);
      log.error(error?.stack || error?.message || String(error));
    } catch (e) {
      void 0;
    }
    throw error;
  }
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
    try {
      log.error('Error getting video metadata:', videoFilePath);
      log.error(error?.stack || error?.message || String(error));
    } catch (e) {
      void 0;
    }
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
    try {
      log.error('Error generating video thumbnail:', thumbnailFullPath);
      log.error(error?.stack || error?.message || String(error));
    } catch (e) {
      void 0;
    }
    throw error;
  }
}

// mp4HasFrames scans top-level MP4 boxes for an mdat with a non-empty
// payload. ffmpeg exits 0 but encodes zero frames when the seek lands after
// the last frame (e.g. single-frame GIFs), leaving a frameless mp4 behind.
async function mp4HasFrames(filePath) {
  let buf;
  try {
    buf = await fs.promises.readFile(filePath);
  } catch (_) {
    return false;
  }
  let offset = 0;
  while (offset + 8 <= buf.length) {
    let size = buf.readUInt32BE(offset);
    const boxType = buf.toString('ascii', offset + 4, offset + 8);
    let payload = size - 8;
    if (size === 0) {
      // box extends to end of file
      size = buf.length - offset;
      payload = size - 8;
    } else if (size === 1) {
      // 64-bit largesize follows the type
      if (offset + 16 > buf.length) return false;
      size = Number(buf.readBigUInt64BE(offset + 8));
      payload = size - 16;
    }
    if (boxType === 'mdat' && payload > 0) return true;
    if (size < 8) return false;
    offset += size;
  }
  return false;
}

async function createVideoThumbnail(
  videoFilePath,
  thumbnailFullPath,
  timeStamp
) {
  try {
    const metadata = await getVideoMetadata(videoFilePath);
    const duration_sec = Number(metadata.format.duration) || 0;
    let thumbnailTime = timeStamp || duration_sec / 2;
    // Seeking near the end of a very short file (e.g. a single-frame GIF,
    // duration ~0.04s) lands after the only frame and encodes nothing —
    // take the first frame instead.
    if (
      duration_sec > 0 &&
      (duration_sec < 1 || thumbnailTime >= duration_sec)
    ) {
      thumbnailTime = 0;
    }
    const useMiddle = duration_sec > 6;

    await generateVideoThumbnail(
      videoFilePath,
      thumbnailTime,
      thumbnailFullPath,
      useMiddle
    );
    if (!(await mp4HasFrames(thumbnailFullPath))) {
      if (thumbnailTime > 0) {
        await generateVideoThumbnail(videoFilePath, 0, thumbnailFullPath, false);
      }
      if (!(await mp4HasFrames(thumbnailFullPath))) {
        try {
          await fs.promises.unlink(thumbnailFullPath);
        } catch (e) {
          void 0;
        }
        throw new Error('ffmpeg produced an empty video thumbnail');
      }
    }
  } catch (err) {
    try {
      log.error('Error during thumbnail generation', videoFilePath);
      log.error(err?.stack || err?.message || String(err));
    } catch (e) {
      void 0;
    }
    throw err;
  }
}

workerpool.worker({
  createThumbnail: createThumbnail,
});

// Ensure unexpected errors get logged
process.on('uncaughtException', (err) => {
  try {
    log.error(
      'UncaughtException in image-processing-worker:',
      err?.stack || err?.message || String(err)
    );
  } catch (_) {
    void 0;
  }
  process.exit(1);
});
process.on('unhandledRejection', (reason) => {
  try {
    log.error('UnhandledRejection in image-processing-worker:', reason);
  } catch (_) {
    void 0;
  }
  process.exit(1);
});
