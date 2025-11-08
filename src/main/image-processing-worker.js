const fs = require('fs');
const sharp = require('sharp');
const path = require('path');
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
const thumbLogPath = process.env.THUMB_LOG || '';
function fileLog(...args) {
  try {
    if (!thumbLogPath) return;
    const line = `[${new Date().toISOString()}] ${args
      .map((a) => (typeof a === 'string' ? a : JSON.stringify(a)))
      .join(' ')}\n`;
    fs.appendFileSync(thumbLogPath, line);
  } catch (_) {
    // best-effort
  }
}

// Reduce libvips caching and limit concurrency to avoid rare assertion failures
try {
  sharp.cache({ files: 0, memory: 0, items: 0 });
  sharp.concurrency(1);
} catch (_) {
  try {
    log.error('Error configuring sharp:', _);
  } catch (e) {
    void 0;
  }
  fileLog('Error configuring sharp', String(_?.message || _));
}

const isDev = process.env.NODE_ENV === 'development';
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
  // svg is conditionally supported; see getFileType
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
  fileLog('createThumbnail:start', { filePath, basePath, cache, timeStamp });
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
    fileLog('createThumbnail:unsupported', { filePath });
    throw new Error('Unsupported file type');
  }

  fileLog('createThumbnail:done', { out: thumbnailFullPath });
  return thumbnailFullPath;
}

async function createImageThumbnail(filePath, thumbnailFullPath, cache) {
  fileLog('createImageThumbnail:start', { filePath, out: thumbnailFullPath });
  const ext = path.extname(filePath).toLowerCase();
  const supportsSvgInput =
    sharp.format && sharp.format.svg && sharp.format.svg.input;
  if (ext === '.svg' && !supportsSvgInput) {
    fileLog('createImageThumbnail:no-svg-support');
    throw new Error('SVG input not supported by current libvips build');
  }
  const stat = await fs.promises.stat(filePath);
  if (!stat.isFile() || stat.size === 0) {
    fileLog('createImageThumbnail:invalid-file');
    throw new Error('Invalid or empty image file');
  }
  const targetSize = cacheSizes[cache] || 600;
  try {
    await sharp(filePath).resize(targetSize).webp().toFile(thumbnailFullPath);
    fileLog('createImageThumbnail:sharp-ok');
  } catch (err) {
    try {
      log.error(
        'Sharp image thumbnail failed, falling back to ffmpeg',
        JSON.stringify({ filePath, thumbnailFullPath, targetSize })
      );
      log.error(err?.stack || err?.message || String(err));
    } catch (e) {
      void 0;
    }
    fileLog('createImageThumbnail:sharp-failed', String(err?.message || err));
    // Fallback to ffmpeg-based conversion for images if sharp fails
    await generateImageThumbnailWithFFmpeg(
      filePath,
      thumbnailFullPath,
      targetSize
    );
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
    fileLog('getVideoMetadata:error', String(error?.message || error));
    throw error;
  }
}

async function generateImageThumbnailWithFFmpeg(
  imageFilePath,
  thumbnailFullPath,
  targetSize
) {
  fileLog('generateImageThumbnailWithFFmpeg:start', {
    imageFilePath,
    out: thumbnailFullPath,
    targetSize,
  });
  const scaleExpr = `scale='min(${targetSize},iw)':-2:force_original_aspect_ratio=decrease`;
  const ffmpegArgs = [
    '-y',
    '-i',
    imageFilePath,
    '-vf',
    scaleExpr,
    '-frames:v',
    '1',
    thumbnailFullPath,
  ];
  try {
    await execFile(ffmpegPath, ffmpegArgs);
  } catch (error) {
    try {
      log.error('FFmpeg image fallback failed:', imageFilePath);
      log.error(error?.stack || error?.message || String(error));
    } catch (e) {
      void 0;
    }
    fileLog(
      'generateImageThumbnailWithFFmpeg:error',
      String(error?.message || error)
    );
    throw error;
  }
  fileLog('generateImageThumbnailWithFFmpeg:done');
}

async function generateVideoThumbnail(
  videoFilePath,
  thumbnailTime,
  thumbnailFullPath,
  useMiddle
) {
  fileLog('generateVideoThumbnail:start', {
    videoFilePath,
    out: thumbnailFullPath,
    t: thumbnailTime,
  });
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
    fileLog('generateVideoThumbnail:error', String(error?.message || error));
    throw error;
  }
  fileLog('generateVideoThumbnail:done');
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
    try {
      log.error('Error during thumbnail generation', videoFilePath);
      log.error(err?.stack || err?.message || String(err));
    } catch (e) {
      void 0;
    }
    fileLog('createVideoThumbnail:error', String(err?.message || err));
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
  fileLog('uncaughtException', String(err?.stack || err?.message || err));
  process.exit(1);
});
process.on('unhandledRejection', (reason) => {
  try {
    log.error('UnhandledRejection in image-processing-worker:', reason);
  } catch (_) {
    void 0;
  }
  fileLog('unhandledRejection', String(reason));
  process.exit(1);
});
process.on('exit', (code) => {
  fileLog('worker:exit', String(code));
});
